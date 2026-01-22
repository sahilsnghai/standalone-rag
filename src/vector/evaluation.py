import time
from typing import Any, Dict, List, Union

from utils.logger import get_logger

logger = get_logger()

from langchain_core.documents import Document


class RAGEvaluator:
    """Evaluate RAG pipeline performance and answer quality.

    Supports both ``Document`` objects and plain dictionaries. The evaluator
    computes latency, average relevance score (if present), a simple
    faithfulness heuristic, and a completeness check.
    """

    @staticmethod
    def _to_dict(doc: Union[Document, Dict[str, Any]]) -> Dict[str, Any]:
        """Convert a ``Document`` or dict to a uniform dict representation.

        The returned dict contains at least ``content`` and ``metadata`` keys.
        If the original object has a ``score`` attribute, it is included.
        """
        if isinstance(doc, Document):
            base = {"content": doc.page_content, "metadata": doc.metadata}
            # Some rerankers may attach a ``score`` attribute to Document.
            if hasattr(doc, "score"):
                base["score"] = getattr(doc, "score")
            return base
        return doc

    @staticmethod
    def _average_score(docs: List[Dict[str, Any]]) -> float | None:
        scores = [
            d.get("score") for d in docs if isinstance(d.get("score"), (int, float))
        ]
        if not scores:
            return None
        return sum(scores) / len(scores)

    @staticmethod
    def _faithfulness_check(answer: str, docs: List[Dict[str, Any]]) -> bool:
        if not answer:
            return False
        context = " ".join(d.get("content", "") for d in docs)
        words = answer.split()
        for n in range(4, min(8, len(words) + 1)):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i : i + n])
                if phrase in context:
                    return True
        return False

    @staticmethod
    def _completeness_check(answer: str, min_len: int = 30) -> bool:
        return len(answer.strip()) >= min_len

    def evaluate(
        self,
        query: str,
        docs: List[Union[Document, Dict[str, Any]]],
        answer: str,
        start_time: float,
    ) -> Dict[str, Any]:
        """Run evaluation metrics on the provided query and documents.

        Parameters
        ----------
        query: str
            The original user query.
        docs: List[Document] | List[dict]
            Retrieved documents.
        answer: str
            The generated answer.
        start_time: float
            Timestamp captured before the pipeline started.
        """
        logger.info(f"Starting evaluation for query: '{query}'")
        # Normalise documents to dicts for internal processing.
        docs_dicts = [self._to_dict(d) for d in docs]
        latency = time.time() - start_time
        avg_score = self._average_score(docs_dicts)
        faithfulness = self._faithfulness_check(answer, docs_dicts)
        completeness = self._completeness_check(answer)

        result = {
            "query": query,
            "retrieval_chunks": len(docs_dicts),
            "average_score": avg_score,
            "latency": latency,
            "faithfulness": faithfulness,
            "completeness": completeness,
        }
        logger.info(f"Evaluation result: {result}")
        return result

evaluator = RAGEvaluator()
