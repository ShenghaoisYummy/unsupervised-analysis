import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION")
    
    # LightRAG Configuration
    RAG_WORKING_DIR = os.getenv("RAG_WORKING_DIR", "./rag_storage")
    
    # Analysis Configuration
    MAX_TOPICS_PER_ARTICLE = int(os.getenv("MAX_TOPICS_PER_ARTICLE", "10"))
    CLUSTERING_THRESHOLD = float(os.getenv("CLUSTERING_THRESHOLD", "0.7"))
    
    # Directory paths
    DOCUMENTS_DIR = "./documents"
    INTERVIEWS_DIR = "./interviews"
    RESULTS_DIR = "./results"
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        return True