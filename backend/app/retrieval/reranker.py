"""
Cross-encoder reranking for improving retrieval precision.
Uses sentence-transformers cross-encoder models.
"""
from typing import List, Tuple
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranker for precise query-document matching.
    """

    def __init__(self, model_name: str | None = None):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: Cross-encoder model name (defaults to config)
        """
        self.model_name = model_name or settings.cross_encoder_model

        logger.info("Loading cross-encoder model", model=self.model_name)

        try:
            self.model = CrossEncoder(self.model_name, max_length=512)
            logger.info("Cross-encoder model loaded successfully")
        except Exception as e:
            logger.error(
                "Failed to load cross-encoder model",
                model=self.model_name,
                error=str(e)
            )
            raise

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int | None = None
    ) -> List[Document]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Search query
            documents: Candidate documents
            top_k: Number of top documents to return

        Returns:
            Reranked documents (top_k)
        """
        if not documents:
            return []

        top_k = top_k or settings.rerank_top_k

        if len(documents) <= top_k:
            logger.debug("Rerank: fewer documents than top_k, scoring all")
            top_k = len(documents)

        logger.debug(
            "Reranking documents",
            query_preview=query[:100],
            num_docs=len(documents),
            top_k=top_k
        )

        try:
            # Prepare query-document pairs
            pairs = [(query, doc.page_content) for doc in documents]

            # Get cross-encoder scores
            scores = self.model.predict(pairs)

            # Create (document, score) tuples
            doc_scores: List[Tuple[Document, float]] = list(zip(documents, scores))

            # Sort by score (descending)
            doc_scores.sort(key=lambda x: x[1], reverse=True)

            # Take top_k
            top_docs = doc_scores[:top_k]

            # Add rerank scores to metadata
            reranked_docs = []
            for doc, score in top_docs:
                doc.metadata["rerank_score"] = float(score)
                reranked_docs.append(doc)

            logger.info(
                "Reranking completed",
                input_docs=len(documents),
                output_docs=len(reranked_docs),
                avg_score=round(sum(s for _, s in top_docs) / len(top_docs), 3) if top_docs else 0
            )

            return reranked_docs

        except Exception as e:
            logger.error(
                "Reranking failed",
                error=str(e),
                num_docs=len(documents)
            )
            # Fallback: return original documents truncated
            logger.warning("Falling back to original document order")
            return documents[:top_k]

    async def arerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int | None = None
    ) -> List[Document]:
        """
        Async version of rerank (runs in executor).

        Args:
            query: Search query
            documents: Candidate documents
            top_k: Number of top documents

        Returns:
            Reranked documents
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                self.rerank,
                query,
                documents,
                top_k
            )
        return result


# Global reranker instance (singleton)
_reranker: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    """
    Get global reranker instance (singleton).

    Returns:
        CrossEncoderReranker instance
    """
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoderReranker()

    return _reranker


async def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int | None = None
) -> List[Document]:
    """
    Convenience function for reranking documents.

    Args:
        query: Search query
        documents: Documents to rerank
        top_k: Number of top documents

    Returns:
        Reranked documents
    """
    reranker = get_reranker()
    return await reranker.arerank(query, documents, top_k)
