from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.config import Config


class EmbeddingModel:
    def __init__(self):
        model_name = Config.EMBEDDING_MODEL

        if model_name.startswith("text-embedding") or model_name.startswith("openai"):
            self.model = OpenAIEmbeddings(model=model_name)

        elif model_name in {
            "all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
        }:
            self.model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        else:
            raise ValueError(f"Unsupported embedding model: {model_name}")

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self.model.aembed_documents(texts)

    async def aembed_query(self, query: str) -> List[float]:
        return await self.model.aembed_query(query)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        return self.model.embed_query(query)
