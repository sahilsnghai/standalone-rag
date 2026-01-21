from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from utils.config import Config
from utils.logger import get_logger

logger = get_logger()


class EmbeddingModel:
    def __init__(self):
        try:
            model_name = Config.EMBEDDING_MODEL
            logger.info(f"Embedding model name: {model_name}")

            if model_name.startswith("text-embedding") or model_name.startswith(
                "openai"
            ):
                self.model = OpenAIEmbeddings(model=model_name)
                self.size = 1536

            elif model_name in {
                "all-MiniLM-L6-v2",
                "sentence-transformers/all-MiniLM-L6-v2",
            }:
                self.model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                self.size = 384

            else:
                raise ValueError(f"Unsupported embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing EmbeddingModel: {e}")
            raise

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Embedding {len(texts)} documents...")
        embeddings = await self.model.aembed_documents(texts)
        logger.info(f"Generated {len(embeddings)} document embeddings.")
        return embeddings

    async def aembed_query(self, query: str) -> List[float]:
        logger.info(f"Embedding query: '{query}'")
        embedding = await self.model.aembed_query(query)
        logger.info("Query embedding generated.")
        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Embedding {len(texts)} documents (sync)...")
        embeddings = self.model.embed_documents(texts)
        logger.info(f"Generated {len(embeddings)} document embeddings (sync).")
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        logger.info(f"Embedding query (sync): '{query}'")
        embedding = self.model.embed_query(query)
        logger.info("Query embedding generated (sync).")
        return embedding

embedder = EmbeddingModel()
