import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import Config

class WithinArticleTopicClustering:
    def __init__(self):
        self.config = Config()
        self.config.validate()
        
        # Set OpenAI API key
        openai.api_key = self.config.OPENAI_API_KEY
        if self.config.OPENAI_ORGANIZATION:
            openai.organization = self.config.OPENAI_ORGANIZATION
    
    async def extract_topics_with_gpt(self, article_content: str, filename: str) -> Dict[str, Any]:
        """Use GPT-4 to extract key topics from an article"""
        
        prompt = f"""
Analyze the following interview text and identify the key topics discussed. 
For each topic, provide:
1. Topic name (2-4 words)
2. Brief description (1-2 sentences)
3. Key quotes or phrases that represent this topic
4. Importance score (1-10, where 10 is most important)

Please identify up to {self.config.MAX_TOPICS_PER_ARTICLE} distinct topics.

Article content:
{article_content[:8000]}  # Limit content to avoid token limits

Respond in JSON format:
{{
  "topics": [
    {{
      "name": "Topic Name",
      "description": "Brief description of the topic",
      "key_phrases": ["phrase1", "phrase2", "phrase3"],
      "importance": 8,
      "supporting_quotes": ["quote1", "quote2"]
    }}
  ]
}}
"""

        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert qualitative researcher analyzing interview data. Provide accurate, insightful topic extraction in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON, handle potential formatting issues
            try:
                # Remove markdown code blocks if present
                if content.startswith("```json"):
                    content = content[7:-3]
                elif content.startswith("```"):
                    content = content[3:-3]
                
                result = json.loads(content)
                
                # Add metadata
                result["filename"] = filename
                result["word_count"] = len(article_content.split())
                result["extraction_method"] = "gpt-4"
                
                return result
                
            except json.JSONDecodeError as e:
                print(f"✗ JSON parsing failed for {filename}: {e}")
                print(f"GPT Response: {content[:200]}...")
                return {"topics": [], "filename": filename, "error": "json_parse_failed"}
                
        except Exception as e:
            print(f"✗ GPT extraction failed for {filename}: {e}")
            return {"topics": [], "filename": filename, "error": str(e)}
    
    def cluster_topics_within_article(self, topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster similar topics within a single article using TF-IDF and cosine similarity"""
        
        if len(topics) <= 2:
            # No need to cluster if we have very few topics
            return topics
        
        # Combine topic information for clustering
        topic_texts = []
        for topic in topics:
            combined_text = f"{topic['name']} {topic['description']} {' '.join(topic.get('key_phrases', []))}"
            topic_texts.append(combined_text)
        
        # Use TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform(topic_texts)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Group similar topics (threshold-based clustering)
            clustered_topics = []
            used_indices = set()
            
            for i, topic in enumerate(topics):
                if i in used_indices:
                    continue
                
                # Find similar topics
                similar_indices = [i]
                for j in range(i + 1, len(topics)):
                    if j not in used_indices and similarity_matrix[i][j] > self.config.CLUSTERING_THRESHOLD:
                        similar_indices.append(j)
                        used_indices.add(j)
                
                used_indices.add(i)
                
                if len(similar_indices) > 1:
                    # Merge similar topics
                    merged_topic = self.merge_topics([topics[idx] for idx in similar_indices])
                    clustered_topics.append(merged_topic)
                else:
                    # Keep topic as is
                    clustered_topics.append(topic)
            
            return clustered_topics
            
        except Exception as e:
            print(f"Clustering failed: {e}")
            return topics  # Return original topics if clustering fails
    
    def merge_topics(self, similar_topics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge similar topics into a single topic"""
        
        # Combine names
        names = [topic['name'] for topic in similar_topics]
        merged_name = " / ".join(names[:2])  # Use first two names
        
        # Combine descriptions
        descriptions = [topic['description'] for topic in similar_topics]
        merged_description = " | ".join(descriptions)
        
        # Combine key phrases
        all_phrases = []
        for topic in similar_topics:
            all_phrases.extend(topic.get('key_phrases', []))
        merged_phrases = list(set(all_phrases))  # Remove duplicates
        
        # Average importance
        importances = [topic.get('importance', 5) for topic in similar_topics]
        merged_importance = round(sum(importances) / len(importances))
        
        # Combine supporting quotes
        all_quotes = []
        for topic in similar_topics:
            all_quotes.extend(topic.get('supporting_quotes', []))
        merged_quotes = list(set(all_quotes))[:3]  # Keep top 3 unique quotes
        
        return {
            "name": merged_name,
            "description": merged_description,
            "key_phrases": merged_phrases,
            "importance": merged_importance,
            "supporting_quotes": merged_quotes,
            "merged_from": names,
            "cluster_size": len(similar_topics)
        }
    
    async def process_single_article(self, filepath: str) -> Dict[str, Any]:
        """Process a single article for topic extraction and clustering"""
        
        filename = Path(filepath).name
        print(f"Processing: {filename}")
        
        try:
            # Load article content
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract topics using GPT
            extraction_result = await self.extract_topics_with_gpt(content, filename)
            
            if extraction_result.get('error'):
                return extraction_result
            
            topics = extraction_result.get('topics', [])
            
            if not topics:
                print(f"✗ No topics extracted for {filename}")
                return {"filename": filename, "topics": [], "clustered_topics": []}
            
            # Cluster similar topics within the article
            clustered_topics = self.cluster_topics_within_article(topics)
            
            result = {
                "filename": filename,
                "filepath": filepath,
                "word_count": extraction_result.get('word_count', 0),
                "original_topics": topics,
                "clustered_topics": clustered_topics,
                "topics_count": len(topics),
                "clustered_count": len(clustered_topics),
                "reduction_ratio": round((len(topics) - len(clustered_topics)) / len(topics), 2) if topics else 0
            }
            
            print(f"✓ {filename}: {len(topics)} → {len(clustered_topics)} topics")
            return result
            
        except Exception as e:
            print(f"✗ Failed to process {filename}: {e}")
            return {"filename": filename, "error": str(e)}
    
    async def process_all_articles(self) -> List[Dict[str, Any]]:
        """Process all articles in the interviews directory"""
        
        interviews_path = Path(self.config.INTERVIEWS_DIR)
        if not interviews_path.exists():
            raise FileNotFoundError(f"Interviews directory not found: {interviews_path}")
        
        markdown_files = list(interviews_path.glob("*.md"))
        if not markdown_files:
            raise FileNotFoundError(f"No markdown files found in {interviews_path}")
        
        print(f"Stage 1: Processing {len(markdown_files)} articles for topic clustering...")
        
        results = []
        for filepath in markdown_files:
            result = await self.process_single_article(str(filepath))
            results.append(result)
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(1)
        
        # Save results
        self.save_stage1_results(results)
        
        return results
    
    def save_stage1_results(self, results: List[Dict[str, Any]]):
        """Save Stage 1 results to JSON file"""
        
        os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
        output_file = os.path.join(self.config.RESULTS_DIR, "stage1_within_article_topics.json")
        
        # Create summary
        summary = {
            "stage": 1,
            "description": "Within-article topic clustering",
            "total_articles": len(results),
            "successful_articles": len([r for r in results if not r.get('error')]),
            "failed_articles": len([r for r in results if r.get('error')]),
            "total_original_topics": sum(r.get('topics_count', 0) for r in results),
            "total_clustered_topics": sum(r.get('clustered_count', 0) for r in results),
            "articles": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Stage 1 results saved to: {output_file}")
        print(f"Summary: {summary['total_original_topics']} → {summary['total_clustered_topics']} topics across {summary['successful_articles']} articles")

async def main():
    """Main function for standalone execution"""
    
    clustering = WithinArticleTopicClustering()
    
    try:
        results = await clustering.process_all_articles()
        
        # Print summary
        successful = [r for r in results if not r.get('error')]
        failed = [r for r in results if r.get('error')]
        
        print(f"\n=== Stage 1 Complete ===")
        print(f"Successful: {len(successful)}/{len(results)} articles")
        print(f"Total topics extracted: {sum(r.get('topics_count', 0) for r in successful)}")
        print(f"Total topics after clustering: {sum(r.get('clustered_count', 0) for r in successful)}")
        
        if failed:
            print(f"Failed articles: {len(failed)}")
            for failure in failed:
                print(f"  - {failure['filename']}: {failure.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"Stage 1 execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())