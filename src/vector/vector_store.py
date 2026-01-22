from __future__ import annotations

import uuid
from typing import Any, Dict, List

from src.vector.embedding import EmbeddingModel, embedder
from utils.logger import get_logger

logger = get_logger()

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


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
    ) -> None:
        try:
            logger.info(f"Ensuring collection exists: {collection_name}")
            exists = await self.client.collection_exists(collection_name)
            if not exists:
                logger.info(f"Collection {collection_name} does not exist. Creating...")
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.embedder.size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created collection: {collection_name}")
                return True
            logger.info(f"Collection {collection_name} already exists.")
        except Exception as e:
            logger.error(f"Error ensuring collection {collection_name}: {e}")
            raise

    async def add(
        self,
        collection_name: str,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        try:
            logger.info(
                f"Starting add operation for collection: {collection_name} with {len(texts)} texts"
            )
            if len(texts) != len(metadatas):
                raise ValueError("texts and metadatas must have the same length")

            logger.info("Generating embeddings for documents...")
            embeddings = await self.embedder.aembed_documents(texts)
            logger.info("Embeddings generated successfully.")

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[i],
                    payload={**metadatas[i], "text": texts[i]},
                )
                for i in range(len(texts))
            ]

            logger.info(f"Upserting {len(points)} points to Qdrant...")
            await self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )
            logger.info(
                f"Successfully added {len(points)} points to collection {collection_name}"
            )
        except Exception as e:
            logger.error(f"Error adding to collection {collection_name}: {e}")
            raise

    async def similarity_search(
        self,
        collection_name: str,
        query: str,
        k: int = 5,
    ):
        try:
            logger.info(
                f"Starting similarity search in {collection_name} for query: '{query}' with k={k}"
            )
            query_embedding = await self.embedder.aembed_query(query)
            logger.info("Query embedding generated.")

            results = await self.client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=k,
                with_payload=True,
            )
            logger.info(
                f"Similarity search complete. Found {len(results.points)} results."
            )
            return results
        except Exception as e:
            logger.error(
                f"Error in similarity search for collection {collection_name}: {e}"
            )
            raise

    async def delete_collection(
        self,
        collection_name: str,
    ) -> None:
        try:
            logger.info(f"Attempting to delete collection: {collection_name}")
            result = await self.client.delete_collection(
                collection_name=collection_name
            )
            logger.info(f"Delete collection result: {result}")
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            raise

vector_store = VectorStore(embedder)