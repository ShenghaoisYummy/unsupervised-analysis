#!/usr/bin/env python3
"""
Main Analysis Orchestrator
Coordinates the complete interview analysis pipeline:
1. Document parsing (Word to Markdown)
2. Knowledge base creation (LightRAG)
3. Stage 1: Within-article topic clustering
4. Stage 2: Cross-article theme clustering
"""

import asyncio
import sys
import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from docx_parser import DocxToMarkdownParser
from lightrag_setup import LightRAGKnowledgeBase
from stage1_clustering import WithinArticleTopicClustering
from stage2_clustering import CrossArticleThemeClustering

class InterviewAnalyzer:
    def __init__(self):
        self.config = Config()
        self.start_time = time.time()
        
        # Initialize components
        self.docx_parser = DocxToMarkdownParser()
        self.knowledge_base = LightRAGKnowledgeBase()
        self.stage1_clustering = WithinArticleTopicClustering()
        self.stage2_clustering = CrossArticleThemeClustering()
    
    def print_banner(self):
        """Print welcome banner"""
        print("="*60)
        print("🔍 INTERVIEW ANALYSIS SYSTEM")
        print("   Unsupervised Analysis with LightRAG & OpenAI")
        print("="*60)
        print()
    
    def validate_environment(self) -> bool:
        """Validate that all required configurations are present"""
        try:
            self.config.validate()
            print("✓ Environment configuration validated")
            return True
        except Exception as e:
            print(f"✗ Environment validation failed: {e}")
            print("\nPlease ensure you have:")
            print("1. Created .env file with your OPENAI_API_KEY")
            print("2. Copied .env.template to .env and filled in your API key")
            return False
    
    def check_input_files(self) -> Dict[str, Any]:
        """Check what input files are available"""
        
        status = {
            "docx_files": [],
            "md_files": [],
            "has_docx": False,
            "has_md": False,
            "needs_conversion": False
        }
        
        # Check for Word documents
        documents_path = Path(self.config.DOCUMENTS_DIR)
        if documents_path.exists():
            docx_files = list(documents_path.glob("*.docx")) + list(documents_path.glob("*.doc"))
            status["docx_files"] = [str(f) for f in docx_files]
            status["has_docx"] = len(docx_files) > 0
        
        # Check for existing markdown files
        interviews_path = Path(self.config.INTERVIEWS_DIR)
        if interviews_path.exists():
            md_files = list(interviews_path.glob("*.md"))
            status["md_files"] = [str(f) for f in md_files]
            status["has_md"] = len(md_files) > 0
        
        # Determine if conversion is needed
        status["needs_conversion"] = status["has_docx"] and not status["has_md"]
        
        return status
    
    async def run_document_conversion(self) -> bool:
        """Step 1: Convert Word documents to Markdown"""
        print("\n" + "="*50)
        print("📄 STEP 1: DOCUMENT CONVERSION")
        print("="*50)
        
        try:
            converted_files = self.docx_parser.convert_batch()
            
            if converted_files:
                print(f"✓ Successfully converted {len(converted_files)} documents")
                return True
            else:
                print("✗ No documents were converted")
                return False
                
        except Exception as e:
            print(f"✗ Document conversion failed: {e}")
            return False
    
    async def run_knowledge_base_creation(self) -> bool:
        """Step 2: Create LightRAG Knowledge Base"""
        print("\n" + "="*50)
        print("🧠 STEP 2: KNOWLEDGE BASE CREATION")
        print("="*50)
        
        try:
            success = await self.knowledge_base.build_knowledge_base()
            
            if success:
                # Show stats
                stats = self.knowledge_base.get_stats()
                print(f"✓ Knowledge base created successfully")
                print(f"  Storage size: {stats.get('storage_size_mb', 0)} MB")
                return True
            else:
                print("✗ Knowledge base creation failed")
                return False
                
        except Exception as e:
            print(f"✗ Knowledge base creation failed: {e}")
            return False
    
    async def run_stage1_analysis(self) -> bool:
        """Step 3: Stage 1 Analysis - Within-article topic clustering"""
        print("\n" + "="*50)
        print("🎯 STEP 3: STAGE 1 - WITHIN-ARTICLE TOPICS")
        print("="*50)
        
        try:
            results = await self.stage1_clustering.process_all_articles()
            
            successful = [r for r in results if not r.get('error')]
            if successful:
                total_topics = sum(r.get('topics_count', 0) for r in successful)
                clustered_topics = sum(r.get('clustered_count', 0) for r in successful)
                
                print(f"✓ Stage 1 completed successfully")
                print(f"  Articles processed: {len(successful)}/{len(results)}")
                print(f"  Topics extracted: {total_topics}")
                print(f"  Topics after clustering: {clustered_topics}")
                return True
            else:
                print("✗ Stage 1 analysis failed - no successful results")
                return False
                
        except Exception as e:
            print(f"✗ Stage 1 analysis failed: {e}")
            return False
    
    async def run_stage2_analysis(self) -> bool:
        """Step 4: Stage 2 Analysis - Cross-article theme clustering"""
        print("\n" + "="*50)
        print("🌐 STEP 4: STAGE 2 - CROSS-ARTICLE THEMES")
        print("="*50)
        
        try:
            results = await self.stage2_clustering.process_stage2_clustering()
            
            if results.get('error'):
                print(f"✗ Stage 2 analysis failed: {results['error']}")
                return False
            
            print(f"✓ Stage 2 completed successfully")
            print(f"  Articles clustered: {results['articles_with_themes']}")
            print(f"  Number of clusters: {results['n_clusters']}")
            print(f"  Clustering quality: {results['silhouette_score']}")
            
            return True
            
        except Exception as e:
            print(f"✗ Stage 2 analysis failed: {e}")
            return False
    
    def generate_summary_report(self):
        """Generate a summary report of the analysis"""
        print("\n" + "="*50)
        print("📊 ANALYSIS SUMMARY")
        print("="*50)
        
        # Check if results exist
        results_dir = Path(self.config.RESULTS_DIR)
        stage1_file = results_dir / "stage1_within_article_topics.json"
        stage2_file = results_dir / "stage2_cross_article_themes.json"
        
        if stage1_file.exists() and stage2_file.exists():
            print("✓ Complete analysis results available:")
            print(f"  - Stage 1 results: {stage1_file}")
            print(f"  - Stage 2 results: {stage2_file}")
            print(f"  - Knowledge base: {self.config.RAG_WORKING_DIR}")
        else:
            print("⚠ Incomplete analysis results")
        
        # Calculate total time
        total_time = time.time() - self.start_time
        print(f"\nTotal processing time: {total_time:.1f} seconds")
    
    async def run_complete_analysis(self, skip_conversion: bool = False) -> bool:
        """Run the complete analysis pipeline"""
        
        self.print_banner()
        
        # Validate environment
        if not self.validate_environment():
            return False
        
        # Check input files
        file_status = self.check_input_files()
        print(f"Input files detected:")
        print(f"  - Word documents: {len(file_status['docx_files'])}")
        print(f"  - Markdown files: {len(file_status['md_files'])}")
        
        # Step 1: Document conversion (if needed)
        if not skip_conversion and file_status['needs_conversion']:
            if not await self.run_document_conversion():
                return False
        elif not file_status['has_md']:
            print("✗ No markdown files found and no Word documents to convert")
            print("Please place your Word documents in:", self.config.DOCUMENTS_DIR)
            return False
        
        # Step 2: Knowledge base creation
        if not await self.run_knowledge_base_creation():
            return False
        
        # Step 3: Stage 1 analysis
        if not await self.run_stage1_analysis():
            return False
        
        # Step 4: Stage 2 analysis
        if not await self.run_stage2_analysis():
            return False
        
        # Generate summary
        self.generate_summary_report()
        
        print("\n🎉 Analysis completed successfully!")
        return True
    
    async def run_single_stage(self, stage: str) -> bool:
        """Run a single stage of the analysis"""
        
        self.print_banner()
        
        if not self.validate_environment():
            return False
        
        if stage == "convert":
            return await self.run_document_conversion()
        elif stage == "knowledge":
            return await self.run_knowledge_base_creation()
        elif stage == "stage1":
            return await self.run_stage1_analysis()
        elif stage == "stage2":
            return await self.run_stage2_analysis()
        else:
            print(f"✗ Unknown stage: {stage}")
            return False

async def main():
    """Main function with command line argument parsing"""
    
    parser = argparse.ArgumentParser(
        description="Interview Analysis System - Unsupervised analysis with LightRAG & OpenAI"
    )
    
    parser.add_argument(
        "--stage",
        choices=["convert", "knowledge", "stage1", "stage2", "all"],
        default="all",
        help="Which stage to run (default: all)"
    )
    
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip document conversion step (use existing markdown files)"
    )
    
    args = parser.parse_args()
    
    analyzer = InterviewAnalyzer()
    
    try:
        if args.stage == "all":
            success = await analyzer.run_complete_analysis(skip_conversion=args.skip_conversion)
        else:
            success = await analyzer.run_single_stage(args.stage)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())