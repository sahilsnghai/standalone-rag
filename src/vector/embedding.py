from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
import torch

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(prefix="[EMBEDDINGS]")


def get_device():
    if not torch.cuda.is_available():
        return "cpu"
    
    try:
        torch.cuda.get_device_properties(0)
        torch.cuda.synchronize()
        return "cuda"
    except RuntimeError:
        logger.warning("CUDA available but not compatible, falling back to CPU")
        return "cpu"


class EmbeddingError(Exception):
    """Custom exception for embedding errors"""
    pass


class EmbeddingModel:
    def __init__(self):
        try:
            model_name = Config.EMBEDDING_MODEL
            logger.info(f"Embedding model name: {model_name}")

            if model_name.startswith("text-embedding") or model_name.startswith("openai"):
                try:
                    logger.info(f"Initializing OpenAI embeddings: {model_name}")
                    self.model = OpenAIEmbeddings(model=model_name)
                    self.size = 1536
                    self.model_type = "openai"
                    logger.info("OpenAI embeddings initialized successfully")
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "quota" in error_msg.lower() or "insufficient_quota" in error_msg:
                        logger.error(f"OpenAI quota exceeded: {e}")
                        logger.warning("Falling back to local HuggingFace embeddings")
                        self._init_local_embeddings()
                    else:
                        raise EmbeddingError(f"OpenAI embedding initialization failed: {e}")

            elif model_name in {
                "all-MiniLM-L6-v2",
                "sentence-transformers/all-MiniLM-L6-v2",
            }:
                self._init_local_embeddings()

            else:
                raise EmbeddingError(f"Unsupported embedding model: {model_name}")
                
        except EmbeddingError as e:
            logger.error(f"EmbeddingError: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing EmbeddingModel: {e}")
            raise EmbeddingError(f"Failed to initialize embedding model: {e}")

    def _init_local_embeddings(self):
        """Initialize local HuggingFace embeddings as fallback"""
        try:
            device = get_device()
            logger.info(f"Initializing local embeddings on device: {device}")
            
            self.model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': device}
            )
            self.size = 384
            self.model_type = "local"
            logger.info("Local embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize local embeddings: {e}")
            raise EmbeddingError(f"Failed to initialize fallback embeddings: {e}")

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            if not texts:
                logger.warning("No texts provided for embedding")
                return []
                
            logger.info(f"Embedding {len(texts)} documents using {self.model_type}...")
            embeddings = await self.model.aembed_documents(texts)
            logger.info(f"Generated {len(embeddings)} document embeddings.")
            return embeddings
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error embedding documents: {e}")
            
            # Handle OpenAI quota errors
            if "429" in error_msg or "quota" in error_msg.lower() or "insufficient_quota" in error_msg:
                logger.warning("OpenAI quota exceeded, falling back to local embeddings")
                try:
                    self._init_local_embeddings()
                    embeddings = await self.model.aembed_documents(texts)
                    logger.info(f"Generated {len(embeddings)} document embeddings (fallback).")
                    return embeddings
                except Exception as fallback_error:
                    raise EmbeddingError(f"Fallback embedding failed: {fallback_error}")
            else:
                raise EmbeddingError(f"Failed to embed documents: {e}")

    async def aembed_query(self, query: str) -> List[float]:
        try:
            if not query:
                logger.warning("Empty query provided for embedding")
                return []
                
            logger.info(f"Embedding query using {self.model_type}: '{query[:100]}...'")
            embedding = await self.model.aembed_query(query)
            logger.info("Query embedding generated.")
            return embedding
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error embedding query: {e}")
            
            # Handle OpenAI quota errors
            if "429" in error_msg or "quota" in error_msg.lower() or "insufficient_quota" in error_msg:
                logger.warning("OpenAI quota exceeded, falling back to local embeddings")
                try:
                    self._init_local_embeddings()
                    embedding = await self.model.aembed_query(query)
                    logger.info("Query embedding generated (fallback).")
                    return embedding
                except Exception as fallback_error:
                    raise EmbeddingError(f"Fallback embedding failed: {fallback_error}")
            else:
                raise EmbeddingError(f"Failed to embed query: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            if not texts:
                logger.warning("No texts provided for embedding")
                return []
                
            logger.info(f"Embedding {len(texts)} documents (sync) using {self.model_type}...")
            embeddings = self.model.embed_documents(texts)
            logger.info(f"Generated {len(embeddings)} document embeddings (sync).")
            return embeddings
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error embedding documents (sync): {e}")
            
            # Handle OpenAI quota errors
            if "429" in error_msg or "quota" in error_msg.lower() or "insufficient_quota" in error_msg:
                logger.warning("OpenAI quota exceeded, falling back to local embeddings")
                try:
                    self._init_local_embeddings()
                    embeddings = self.model.embed_documents(texts)
                    logger.info(f"Generated {len(embeddings)} document embeddings (sync, fallback).")
                    return embeddings
                except Exception as fallback_error:
                    raise EmbeddingError(f"Fallback embedding failed: {fallback_error}")
            else:
                raise EmbeddingError(f"Failed to embed documents: {e}")

    def embed_query(self, query: str) -> List[float]:
        try:
            if not query:
                logger.warning("Empty query provided for embedding")
                return []
                
            logger.info(f"Embedding query (sync) using {self.model_type}: '{query[:100]}...'")
            embedding = self.model.embed_query(query)
            logger.info("Query embedding generated (sync).")
            return embedding
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error embedding query (sync): {e}")
            
            # Handle OpenAI quota errors
            if "429" in error_msg or "quota" in error_msg.lower() or "insufficient_quota" in error_msg:
                logger.warning("OpenAI quota exceeded, falling back to local embeddings")
                try:
                    self._init_local_embeddings()
                    embedding = self.model.embed_query(query)
                    logger.info("Query embedding generated (sync, fallback).")
                    return embedding
                except Exception as fallback_error:
                    raise EmbeddingError(f"Fallback embedding failed: {fallback_error}")
            else:
                raise EmbeddingError(f"Failed to embed query: {e}")


try:
    embedder = EmbeddingModel()
except EmbeddingError as e:
    logger.error(f"Failed to initialize embedding model: {e}")
    raise   