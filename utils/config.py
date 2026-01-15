import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    # Database configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/rag_db")

    # Vector database configuration
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")

    # Model configurations
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Processing configurations
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    # API configurations
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))

    # UI configurations
    UI_HOST = os.getenv("UI_HOST", "0.0.0.0")
    UI_PORT = int(os.getenv("UI_PORT", "8501"))

    # New configuration values for RAG pipeline
    TOP_K = int(os.getenv("TOP_K", "6"))
    RERANK_K = int(os.getenv("RERANK_K", "6"))
    MAX_DOC_DISPLAY = int(os.getenv("MAX_DOC_DISPLAY", "5"))
