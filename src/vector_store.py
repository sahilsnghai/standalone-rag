from __future__ import annotations

from typing import List, Dict, Any, Callable, Optional
import uuid
from src.embedding import EmbeddingModel

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    VectorParams,
    Distance,
)


class VectorStore:
    """
    Async Qdrant vector store.
    - Embeddings computed explicitly
    - chat_id stored in payload for filtering
    """

    def __init__(
        self,
        embedder: EmbeddingModel,
        host: str = "0.0.0.0",
        port: int = 6333,
    ):
        self.embedder: EmbeddingModel = embedder
        self.client = AsyncQdrantClient(host=host, port=port)

    async def ensure_collection(
        self,
        collection_name: str,
        size: int = 1536,
    ) -> None:
        exists = await self.client.collection_exists(collection_name)
        if not exists:
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=size,
                    distance=Distance.COSINE,
                ),
            )
            return True

    async def add(
        self,
        collection_name: str,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have the same length")

        embeddings = await self.embedder.aembed_documents(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i],
                payload={**metadatas[i], "text": texts[i]},
            )
            for i in range(len(texts))
        ]

        await self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

    async def similarity_search(
        self,
        collection_name: str,
        query: str,
        k: int = 5,
    ):
        query_embedding = await self.embedder.aembed_query(query)

        return await self.client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=k,
            with_payload=True
        )

    async def delete_collection(
        self,
        collection_name: str,
    ) -> None:
        await self.client.delete_collection(
            collection_name=collection_name
        )
