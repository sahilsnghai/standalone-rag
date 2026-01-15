from typing import List, Optional
from langchain_core.documents import Document

from src.reranker import Reranker
from utils.config import Config
from qdrant_client.models import ScoredPoint


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
        if not self.retrieves:
            return []

        # for _, r in self.retrieves:
        #     for i in r:
        #         r: List[ScoredPoint]
        #         print(f"{r=}")

        #         print(f"{i.id=}")
        #         print(f"{i.score=}")
        #         print(f"{i.payload.get("text")=}")
        #         print(f"{i.payload.get("metadata")=}")
        #         print(f"{i.payload.get("file_name")=}")

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

        print(f"{candidate_docs = }")

        reranked_docs = self.reranker.rerank(query, candidate_docs)
        return reranked_docs[: self.top_k]
