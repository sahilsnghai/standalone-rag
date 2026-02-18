from typing import List, Optional

from langchain_core.documents import Document
from qdrant_client.models import ScoredPoint, QueryResponse

from src.vector import Reranker
from utils.config import Config
from utils.logger import get_logger

logger = get_logger(prefix="[RETRIEVAL]")


class RetrievalPipeline:
    """
    Works with already-fetched retrieval results (QueryResponse from Qdrant).
    Each ScoredPoint: {"text": str, "metadata": dict, "score": float, "id": str}
    CPU-compatible: handles CUDA errors gracefully.
    """

    def __init__(
        self,
        retrieves: QueryResponse,
        top_k: Optional[int] = None,
    ):
        self.retrieves = retrieves
        self.top_k = top_k or Config.TOP_K
        logger.info(f"Initializing RetrievalPipeline with top_k={self.top_k}")
        
        try:
            self.reranker = Reranker()
            logger.info("Reranker initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")
            raise

    async def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve and rerank documents for a given query.
        Handles CUDA errors gracefully by falling back to CPU.
        """
        try:
            logger.info(f"Starting retrieval pipeline for query: '{query[:100]}...'")
            logger.debug(f"Full query: {query}")
            
            if not self.retrieves or not self.retrieves.points:
                logger.warning("No retrieval results to process")
                return []

            # Extract ScoredPoints from QueryResponse
            logger.debug(f"Extracting {len(self.retrieves.points)} scored points from QueryResponse")
            scored_points = self.retrieves.points

            # Construct candidate documents
            candidate_docs: List[Document] = []
            
            for scored_point in scored_points:
                try:
                    text_content = (scored_point.payload or {}).get("text") or ""
                    metadata = {
                        **{k: v for k, v in (scored_point.payload or {}).items() if k != "text"},
                        "score": scored_point.score,
                        "id": scored_point.id,
                    }
                    
                    doc = Document(
                        page_content=text_content,
                        metadata=metadata,
                    )
                    candidate_docs.append(doc)
                    logger.debug(f"Added document: ID={scored_point.id}, Score={scored_point.score}")
                except Exception as e:
                    logger.warning(f"Error constructing document from scored point: {e}")
                    continue

            logger.info(f"Candidate docs constructed: {len(candidate_docs)} documents")
            
            if not candidate_docs:
                logger.warning("No valid candidate documents after construction")
                return []

            # Rerank documents
            logger.info(f"Starting reranking of {len(candidate_docs)} documents")
            try:
                reranked_docs = self.reranker.rerank(query, candidate_docs)
                logger.info(f"Reranking complete: {len(reranked_docs)} documents returned")
            except RuntimeError as e:
                # Handle CUDA errors
                if "CUDA" in str(e) or "kernel" in str(e):
                    logger.error(f"CUDA error during reranking: {e}")
                    logger.warning("Falling back to original document order (no reranking)")
                    reranked_docs = candidate_docs
                else:
                    logger.error(f"Runtime error during reranking: {e}")
                    raise
            except Exception as e:
                logger.error(f"Error during reranking: {e}")
                logger.warning("Falling back to original document order")
                reranked_docs = candidate_docs

            # Return top-k
            final_docs = reranked_docs[: self.top_k]
            logger.info(f"Returning top {len(final_docs)} documents (requested top_k={self.top_k})")
            logger.debug(f"Final document IDs: {[doc.metadata.get('id') for doc in final_docs]}")
            
            return final_docs
            
        except Exception as e:
            logger.error(f"Error in retrieve pipeline: {e}")
            logger.warning("Returning empty list due to retrieval pipeline error")
            return []