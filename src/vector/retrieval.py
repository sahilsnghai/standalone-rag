from typing import List, Optional

from langchain_core.documents import Document
from qdrant_client.models import ScoredPoint

from src.vector import Reranker
from utils.config import Config
from utils.logger import get_logger

logger = get_logger()


class RetrievalPipeline:
    """
    Works with already-fetched retrieval results (list of dicts).
    Each dict: {"text": str, "metadata": dict, "score": float, "id": str}
    """

    def __init__(
        self,
        retrieves: List[dict],
        top_k: Optional[int] = None,
    ):
        self.retrieves = retrieves or []
        self.top_k = top_k or Config.TOP_K
        self.reranker = Reranker()

    async def retrieve(self, query: str) -> List[Document]:
        try:
            logger.info(f"Starting retrieval pipeline for query: '{query}'")
            if not self.retrieves:
                logger.warning("No retrieval results to process.")
                return []

            candidate_docs: List[Document] = [
                Document(
                    page_content=(i.payload or {}).get("text") or "",
                    metadata={
                        **{k: v for k, v in (i.payload or {}).items() if k != "text"},
                        "score": i.score,
                        "id": i.id,
                    },
                )
                for _, r in self.retrieves  # unpack (_, r)
                for i in r  # iterate List[ScoredPoint]
            ]

            logger.info(f"Candidate docs constructed: {len(candidate_docs)} docs")
            logger.info(f"Candidate docs: {candidate_docs}")

            logger.info("Starting reranking...")
            reranked_docs = self.reranker.rerank(query, candidate_docs)
            logger.info(f"Reranking complete. Top {self.top_k} docs selected.")

            return reranked_docs[: self.top_k]
        except Exception as e:
            logger.error(f"Error in retrieve: {e}")
            raise



