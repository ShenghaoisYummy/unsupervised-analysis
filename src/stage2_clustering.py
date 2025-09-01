import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from openai import AsyncOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import numpy as np
from config import Config
from lightrag_setup import LightRAGKnowledgeBase

class CrossArticleThemeClustering:
    def __init__(self):
        self.config = Config()
        self.config.validate()
        
        # Set OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.config.OPENAI_API_KEY,
            organization=self.config.OPENAI_ORGANIZATION if self.config.OPENAI_ORGANIZATION else None
        )
        
        # Initialize LightRAG knowledge base
        self.knowledge_base = LightRAGKnowledgeBase()
    
    def load_stage1_results(self) -> Dict[str, Any]:
        """Load Stage 1 results from JSON file"""
        
        stage1_file = os.path.join(self.config.RESULTS_DIR, "stage1_within_article_topics.json")
        
        if not os.path.exists(stage1_file):
            raise FileNotFoundError(f"Stage 1 results not found: {stage1_file}")
        
        with open(stage1_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    async def extract_article_themes_with_gpt(self, article_topics: Dict[str, Any]) -> Dict[str, Any]:
        """Use GPT-4 to extract high-level themes from an article's topics"""
        
        filename = article_topics.get('filename', 'Unknown')
        clustered_topics = article_topics.get('clustered_topics', [])
        
        if not clustered_topics:
            return {"filename": filename, "themes": [], "error": "no_topics"}
        
        # Prepare topics summary for GPT
        topics_summary = []
        for topic in clustered_topics:
            summary = f"- {topic['name']}: {topic['description']} (Importance: {topic.get('importance', 'N/A')})"
            topics_summary.append(summary)
        
        prompt = f"""
Based on the following topics identified in an interview, extract 2-4 high-level themes that represent the overarching subjects discussed in this article.

For each theme, provide:
1. Theme name (2-5 words)
2. Brief description (1-2 sentences)
3. Related topic names from the list below
4. Theme strength (1-10, how prominently this theme appears)

Topics from the interview:
{chr(10).join(topics_summary)}

Respond in JSON format:
{{
  "themes": [
    {{
      "name": "Theme Name",
      "description": "Description of the overarching theme",
      "related_topics": ["topic1", "topic2"],
      "strength": 8,
      "coverage": "how much of the interview this theme covers"
    }}
  ]
}}
"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert qualitative researcher identifying overarching themes from interview topics. Provide accurate, insightful theme extraction in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content.strip()
            
            try:
                # Clean JSON if needed
                if content.startswith("```json"):
                    content = content[7:-3]
                elif content.startswith("```"):
                    content = content[3:-3]
                
                result = json.loads(content)
                result["filename"] = filename
                result["source_topics_count"] = len(clustered_topics)
                
                return result
                
            except json.JSONDecodeError as e:
                print(f"✗ JSON parsing failed for themes in {filename}: {e}")
                return {"filename": filename, "themes": [], "error": "json_parse_failed"}
                
        except Exception as e:
            print(f"✗ GPT theme extraction failed for {filename}: {e}")
            return {"filename": filename, "themes": [], "error": str(e)}
    
    async def query_related_articles_with_lightrag(self, theme_name: str, theme_description: str) -> str:
        """Use LightRAG to find related content across all articles"""
        
        try:
            # Initialize RAG if not already done
            if not self.knowledge_base.rag:
                await self.knowledge_base.initialize_rag()
            
            query = f"Find articles and content related to {theme_name}. Focus on: {theme_description}"
            
            response = await self.knowledge_base.query_knowledge_base(query, mode="hybrid")
            return response
            
        except Exception as e:
            print(f"✗ LightRAG query failed for theme '{theme_name}': {e}")
            return ""
    
    def create_article_theme_vectors(self, articles_with_themes: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
        """Create TF-IDF vectors for articles based on their themes"""
        
        article_texts = []
        article_names = []
        
        for article in articles_with_themes:
            if article.get('error') or not article.get('themes'):
                continue
                
            # Combine all theme information for this article
            theme_texts = []
            for theme in article['themes']:
                combined_text = f"{theme['name']} {theme['description']} {theme.get('coverage', '')}"
                # Weight by strength
                strength = theme.get('strength', 5)
                theme_texts.extend([combined_text] * max(1, strength // 3))
            
            article_text = " ".join(theme_texts)
            article_texts.append(article_text)
            article_names.append(article['filename'])
        
        if not article_texts:
            return np.array([]), []
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=200, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(article_texts)
        
        return tfidf_matrix.toarray(), article_names
    
    def cluster_articles_by_themes(self, articles_with_themes: List[Dict[str, Any]], n_clusters: int = None) -> Dict[str, Any]:
        """Cluster articles based on their themes using K-means"""
        
        # Create vectors
        tfidf_matrix, article_names = self.create_article_theme_vectors(articles_with_themes)
        
        if len(article_names) == 0:
            return {"error": "no_valid_articles"}
        
        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            # Use elbow method or set to reasonable default
            n_articles = len(article_names)
            n_clusters = min(max(2, n_articles // 8), 10)  # Between 2-10 clusters
        
        n_clusters = min(n_clusters, len(article_names))  # Can't have more clusters than articles
        
        try:
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Calculate silhouette score for cluster quality
            from sklearn.metrics import silhouette_score
            if n_clusters > 1:
                silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
            else:
                silhouette_avg = 0
            
            # Create cluster groups
            clusters = {}
            for i, (article_name, label) in enumerate(zip(article_names, cluster_labels)):
                cluster_id = f"cluster_{label}"
                if cluster_id not in clusters:
                    clusters[cluster_id] = {
                        "cluster_id": cluster_id,
                        "cluster_number": int(label),
                        "articles": [],
                        "size": 0
                    }
                
                # Find original article data
                article_data = next((a for a in articles_with_themes if a['filename'] == article_name), None)
                if article_data:
                    clusters[cluster_id]["articles"].append(article_data)
                    clusters[cluster_id]["size"] += 1
            
            return {
                "n_clusters": n_clusters,
                "silhouette_score": round(silhouette_avg, 3),
                "clusters": list(clusters.values()),
                "cluster_centers": kmeans.cluster_centers_.tolist()
            }
            
        except Exception as e:
            print(f"✗ K-means clustering failed: {e}")
            return {"error": str(e)}
    
    async def analyze_cluster_themes_with_gpt(self, cluster: Dict[str, Any]) -> Dict[str, Any]:
        """Use GPT-4 to analyze and summarize themes within a cluster"""
        
        articles = cluster.get('articles', [])
        if not articles:
            return {"error": "no_articles"}
        
        # Gather all themes from articles in this cluster
        all_themes = []
        for article in articles:
            for theme in article.get('themes', []):
                all_themes.append({
                    "article": article['filename'],
                    "theme": theme['name'],
                    "description": theme['description'],
                    "strength": theme.get('strength', 5)
                })
        
        if not all_themes:
            return {"error": "no_themes"}
        
        # Create summary for GPT
        themes_summary = []
        for theme_data in all_themes[:20]:  # Limit to avoid token limits
            themes_summary.append(f"- {theme_data['article']}: {theme_data['theme']} ({theme_data['description']})")
        
        prompt = f"""
Analyze the following themes from {len(articles)} related interview articles and identify:

1. The main overarching theme(s) that unite these articles
2. Common patterns or topics that appear across multiple articles
3. Unique aspects that distinguish this cluster
4. A descriptive cluster name (2-4 words)

Themes from the articles:
{chr(10).join(themes_summary)}

Respond in JSON format:
{{
  "cluster_name": "Descriptive Cluster Name",
  "main_theme": "Primary unifying theme",
  "common_patterns": ["pattern1", "pattern2", "pattern3"],
  "distinguishing_features": ["feature1", "feature2"],
  "cluster_summary": "2-3 sentence summary of what makes this cluster unique"
}}
"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert qualitative researcher analyzing clusters of related interviews. Provide insightful cluster analysis in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            
            try:
                if content.startswith("```json"):
                    content = content[7:-3]
                elif content.startswith("```"):
                    content = content[3:-3]
                
                result = json.loads(content)
                result["articles_count"] = len(articles)
                result["themes_analyzed"] = len(all_themes)
                
                return result
                
            except json.JSONDecodeError as e:
                print(f"✗ JSON parsing failed for cluster analysis: {e}")
                return {"error": "json_parse_failed"}
                
        except Exception as e:
            print(f"✗ GPT cluster analysis failed: {e}")
            return {"error": str(e)}
    
    async def process_stage2_clustering(self) -> Dict[str, Any]:
        """Main Stage 2 processing: Cross-article theme clustering"""
        
        print("Stage 2: Cross-article theme clustering...")
        
        # Load Stage 1 results
        try:
            stage1_data = self.load_stage1_results()
            articles = stage1_data.get('articles', [])
            successful_articles = [a for a in articles if not a.get('error') and a.get('clustered_topics')]
            
            print(f"Loaded {len(successful_articles)} articles with topics from Stage 1")
            
        except Exception as e:
            return {"error": f"Failed to load Stage 1 results: {e}"}
        
        # Extract article-level themes
        print("Extracting article-level themes...")
        articles_with_themes = []
        
        for article in successful_articles:
            theme_result = await self.extract_article_themes_with_gpt(article)
            articles_with_themes.append(theme_result)
            await asyncio.sleep(1)  # Rate limiting
            
            if not theme_result.get('error'):
                print(f"✓ Themes extracted for {theme_result['filename']}: {len(theme_result.get('themes', []))} themes")
        
        # Filter successful theme extractions
        valid_articles = [a for a in articles_with_themes if not a.get('error') and a.get('themes')]
        print(f"Successfully extracted themes from {len(valid_articles)} articles")
        
        if len(valid_articles) < 2:
            return {"error": "Insufficient articles with themes for clustering"}
        
        # Cluster articles by themes
        print("Clustering articles by themes...")
        clustering_result = self.cluster_articles_by_themes(valid_articles)
        
        if clustering_result.get('error'):
            return clustering_result
        
        # Analyze each cluster with GPT
        print("Analyzing cluster themes...")
        analyzed_clusters = []
        
        for cluster in clustering_result['clusters']:
            cluster_analysis = await self.analyze_cluster_themes_with_gpt(cluster)
            
            # Combine cluster data with analysis
            enhanced_cluster = {
                **cluster,
                "analysis": cluster_analysis
            }
            analyzed_clusters.append(enhanced_cluster)
            
            cluster_name = cluster_analysis.get('cluster_name', f"Cluster {cluster['cluster_number']}")
            print(f"✓ Analyzed {cluster_name}: {cluster['size']} articles")
            
            await asyncio.sleep(1)  # Rate limiting
        
        # Create final results
        stage2_results = {
            "stage": 2,
            "description": "Cross-article theme clustering",
            "total_articles": len(articles),
            "articles_with_themes": len(valid_articles),
            "failed_articles": len(articles_with_themes) - len(valid_articles),
            "clustering_method": "k-means",
            "n_clusters": clustering_result['n_clusters'],
            "silhouette_score": clustering_result['silhouette_score'],
            "clusters": analyzed_clusters,
            "all_article_themes": articles_with_themes
        }
        
        # Save results
        self.save_stage2_results(stage2_results)
        
        return stage2_results
    
    def save_stage2_results(self, results: Dict[str, Any]):
        """Save Stage 2 results to JSON file"""
        
        os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
        output_file = os.path.join(self.config.RESULTS_DIR, "stage2_cross_article_themes.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Stage 2 results saved to: {output_file}")

async def main():
    """Main function for standalone execution"""
    
    clustering = CrossArticleThemeClustering()
    
    try:
        results = await clustering.process_stage2_clustering()
        
        if results.get('error'):
            print(f"Stage 2 failed: {results['error']}")
            return
        
        # Print summary
        print(f"\n=== Stage 2 Complete ===")
        print(f"Articles processed: {results['articles_with_themes']}/{results['total_articles']}")
        print(f"Number of clusters: {results['n_clusters']}")
        print(f"Clustering quality (silhouette score): {results['silhouette_score']}")
        
        print(f"\nCluster Summary:")
        for cluster in results['clusters']:
            analysis = cluster.get('analysis', {})
            cluster_name = analysis.get('cluster_name', f"Cluster {cluster['cluster_number']}")
            main_theme = analysis.get('main_theme', 'Unknown theme')
            print(f"  - {cluster_name}: {cluster['size']} articles - {main_theme}")
        
    except Exception as e:
        print(f"Stage 2 execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())