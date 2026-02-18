import os
import warnings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/rag_db")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))

    UI_HOST = os.getenv("UI_HOST", "0.0.0.0")
    UI_PORT = int(os.getenv("UI_PORT", "8501"))

    TOP_K = int(os.getenv("TOP_K", "6"))
    RERANK_K = int(os.getenv("RERANK_K", "6"))
    MAX_DOC_DISPLAY = int(os.getenv("MAX_DOC_DISPLAY", "5"))

    DEBUG = bool(os.getenv("DEBUG", False))
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

    JINA_API_KEY = os.getenv("JINA_API_KEY")
    RERANKER_BACKEND = os.getenv("RERANKER_BACKEND", "jina")

    @classmethod
    def validate_and_warn(cls):
        if not cls.OPENAI_API_KEY:
            warnings.warn(
                "⚠️  OPENAI_API_KEY not set!\n"
                "   Set OPENAI_API_KEY environment variable to use OpenAI embeddings.\n"
                "   Currently using: all-MiniLM-L6-v2 (local model)",
                UserWarning,
                stacklevel=2
            )
        
        if cls.DATABASE_URL == "postgresql://localhost/rag_db":
            warnings.warn(
                "⚠️  Using default PostgreSQL connection: postgresql://localhost/rag_db\n"
                "   Set DATABASE_URL environment variable for production.\n"
                "   Make sure PostgreSQL server is running on localhost:5432",
                UserWarning,
                stacklevel=2
            )
        
        if cls.EMBEDDING_MODEL in {"all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"}:
            import torch
            msg = (
                f"⚠️  Using embedding model: {cls.EMBEDDING_MODEL}\n"
                f"   Model will be downloaded (~90MB) on first use.\n"
            )
            
            if torch.cuda.is_available():
                try:
                    torch.cuda.get_device_properties(0)
                    torch.cuda.synchronize()
                    msg += "   GPU detected: CUDA operations will use GPU ✓"
                except RuntimeError:
                    msg += "   GPU detected but CUDA not compatible: Will fallback to CPU (slower) ⚠️"
            else:
                msg += "   GPU NOT detected: Will use CPU (slower) ⚠️"
            
            warnings.warn(msg, UserWarning, stacklevel=2)


Config.validate_and_warn()