from typing import List, Optional

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from utils.logger import get_logger

logger = get_logger()


# class Reranker:
#     """Neural reranker based on Cohere's cross‑encoder.

#     Parameters
#     ----------
#     model_name: str, optional
#         Name of the Cohere rerank model. Defaults to the value defined in
#         ``Config.RERANKER_MODEL``.
#     top_k: int, optional
#         Number of top documents to return after reranking. Defaults to
#         ``Config.RERANK_K``.
#     """

#     def __init__(self, model_name: str | None = None, top_k: int | None = None):
#         self.model_name = model_name or Config.RERANKER_MODEL
#         self.top_k = top_k or Config.RERANK_K
#         # Initialise the Cohere reranker once; it will be reused for all calls.
#         self.reranker = CohereRerank(model=self.model_name, top_n=self.top_k)

#     def rerank(self, query: str, docs: List[Document]) -> List[Document]:
#         """Rerank ``docs`` for ``query`` using CohereRerank.

#         The ``CohereRerank`` component can accept a list of ``Document`` objects
#         directly. It returns the documents ordered by relevance, already limited
#         to ``top_n``. We therefore simply forward the query and documents.
#         """
#         # CohereRerank expects a list of strings; we extract the content.
#         contents = [doc.page_content for doc in docs]
#         # Perform reranking; the result is a list of indices ordered by relevance.
#         ranked_indices = self.reranker.rerank(query, contents)
#         # Preserve original Document objects in the new order and limit to top_k.
#         reranked_docs = [docs[i] for i in ranked_indices[: self.top_k]]
#         return reranked_docs


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 10,
        device: Optional[str] = None,
    ):
        self.top_k = top_k
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return []

        logger.info(f"Reranking {len(docs)} documents for query: {query}")
        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)

        reranked_docs = [d for d, _ in ranked[: self.top_k]]
        logger.info(f"Reranked to {len(reranked_docs)} documents")
        return reranked_docs
