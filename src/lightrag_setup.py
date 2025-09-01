import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
import openai
from config import Config

class LightRAGKnowledgeBase:
    def __init__(self):
        self.config = Config()
        self.config.validate()
        
        # Set OpenAI API key
        openai.api_key = self.config.OPENAI_API_KEY
        if self.config.OPENAI_ORGANIZATION:
            openai.organization = self.config.OPENAI_ORGANIZATION
        
        # Initialize LightRAG
        self.rag = None
        
    async def initialize_rag(self):
        """Initialize LightRAG instance"""
        try:
            self.rag = LightRAG(
                working_dir=self.config.RAG_WORKING_DIR,
                llm_model_func=openai_complete_if_cache,
                embedding_func=openai_embedding,
            )
            await self.rag.initialize_storages()
            print("✓ LightRAG initialized successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize LightRAG: {e}")
            return False
    
    def load_markdown_files(self) -> List[Dict[str, Any]]:
        """Load all markdown files from interviews directory"""
        interviews_path = Path(self.config.INTERVIEWS_DIR)
        
        if not interviews_path.exists():
            raise FileNotFoundError(f"Interviews directory not found: {interviews_path}")
        
        markdown_files = list(interviews_path.glob("*.md"))
        if not markdown_files:
            raise FileNotFoundError(f"No markdown files found in {interviews_path}")
        
        documents = []
        for file_path in markdown_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                documents.append({
                    'filename': file_path.name,
                    'filepath': str(file_path),
                    'content': content,
                    'word_count': len(content.split())
                })
                
            except Exception as e:
                print(f"✗ Failed to load {file_path}: {e}")
        
        print(f"✓ Loaded {len(documents)} markdown documents")
        return documents
    
    async def insert_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Insert documents into LightRAG knowledge base"""
        if not self.rag:
            raise RuntimeError("LightRAG not initialized. Call initialize_rag() first.")
        
        success_count = 0
        failed_files = []
        
        print(f"Inserting {len(documents)} documents into knowledge base...")
        
        for doc in documents:
            try:
                # Prepare document content with metadata
                content_with_metadata = f"""
Document: {doc['filename']}
Word Count: {doc['word_count']}

{doc['content']}
"""
                
                await self.rag.ainsert(content_with_metadata)
                success_count += 1
                print(f"✓ Inserted: {doc['filename']} ({doc['word_count']} words)")
                
            except Exception as e:
                failed_files.append((doc['filename'], str(e)))
                print(f"✗ Failed to insert {doc['filename']}: {e}")
        
        print(f"\nInsertion Summary:")
        print(f"✓ Successful: {success_count}/{len(documents)}")
        if failed_files:
            print(f"✗ Failed: {len(failed_files)}")
            for filename, error in failed_files:
                print(f"  - {filename}: {error}")
        
        return success_count > 0
    
    async def query_knowledge_base(self, query: str, mode: str = "hybrid") -> str:
        """Query the knowledge base"""
        if not self.rag:
            raise RuntimeError("LightRAG not initialized. Call initialize_rag() first.")
        
        try:
            response = await self.rag.aquery(
                query,
                param=QueryParam(mode=mode)
            )
            return response
        except Exception as e:
            raise Exception(f"Query failed: {e}")
    
    async def build_knowledge_base(self) -> bool:
        """Complete workflow to build knowledge base from markdown files"""
        print("Building LightRAG Knowledge Base...")
        
        # Initialize RAG
        if not await self.initialize_rag():
            return False
        
        # Load documents
        try:
            documents = self.load_markdown_files()
        except Exception as e:
            print(f"✗ Failed to load documents: {e}")
            return False
        
        # Insert documents
        success = await self.insert_documents(documents)
        
        if success:
            print("✓ Knowledge base built successfully!")
            print(f"  - Working directory: {self.config.RAG_WORKING_DIR}")
            print(f"  - Documents processed: {len(documents)}")
            print(f"  - Total words: {sum(doc['word_count'] for doc in documents)}")
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        if not os.path.exists(self.config.RAG_WORKING_DIR):
            return {"status": "not_initialized"}
        
        # Get directory size and file count
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(self.config.RAG_WORKING_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
                file_count += 1
        
        return {
            "status": "initialized",
            "working_dir": self.config.RAG_WORKING_DIR,
            "storage_files": file_count,
            "storage_size_mb": round(total_size / (1024 * 1024), 2)
        }

async def main():
    """Main function for standalone execution"""
    kb = LightRAGKnowledgeBase()
    
    # Build knowledge base
    success = await kb.build_knowledge_base()
    
    if success:
        # Show stats
        stats = kb.get_stats()
        print(f"\nKnowledge Base Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test query
        print("\nTesting knowledge base with sample query...")
        try:
            response = await kb.query_knowledge_base("What are the main themes across all interviews?")
            print(f"Sample response: {response[:200]}...")
        except Exception as e:
            print(f"Test query failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())