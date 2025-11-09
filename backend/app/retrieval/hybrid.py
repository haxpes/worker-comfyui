"""
Hybrid retriever combining BM25 and Dense retrievers.
Uses LangChain's EnsembleRetriever for combination.
"""
from typing import List, Optional
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from app.config import settings
from app.retrieval.bm25 import create_bm25_retriever
from app.retrieval.dense import create_dense_retriever
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PostgresHybridRetriever:
    """
    Hybrid retriever combining BM25 and Dense search.
    Uses LangChain's EnsembleRetriever with equal weights.
    """

    def __init__(
        self,
        bm25_k: int | None = None,
        dense_k: int | None = None,
        weights: List[float] | None = None,
        corpus_id: str | None = None
    ):
        """
        Initialize hybrid retriever.

        Args:
            bm25_k: Number of BM25 results
            dense_k: Number of dense results
            weights: Weights for BM25 and Dense (default [0.5, 0.5])
            corpus_id: Optional corpus filter
        """
        self.bm25_k = bm25_k or settings.bm25_k
        self.dense_k = dense_k or settings.dense_k
        self.weights = weights or [0.5, 0.5]  # Equal weights by default
        self.corpus_id = corpus_id

        # Create retrievers
        self.bm25_retriever = create_bm25_retriever(k=self.bm25_k, corpus_id=corpus_id)
        self.dense_retriever = create_dense_retriever(k=self.dense_k, corpus_id=corpus_id)

        # Create ensemble retriever
        self.ensemble = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.dense_retriever],
            weights=self.weights
        )

        logger.info(
            "Hybrid retriever initialized",
            bm25_k=self.bm25_k,
            dense_k=self.dense_k,
            weights=self.weights,
            corpus_id=corpus_id
        )

    async def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve documents using hybrid search.

        Args:
            query: Search query

        Returns:
            List of documents from both retrievers, combined and ranked
        """
        logger.debug(
            "Hybrid retrieval",
            query_preview=query[:100]
        )

        try:
            # Use EnsembleRetriever's async method
            documents = await self.ensemble.ainvoke(query)

            # Add hybrid retriever tag
            for doc in documents:
                doc.metadata["retriever"] = "hybrid"

            logger.info(
                "Hybrid retrieval completed",
                query_preview=query[:50],
                results=len(documents)
            )

            return documents

        except Exception as e:
            logger.error(
                "Hybrid retrieval failed",
                error=str(e),
                query=query[:100]
            )
            raise

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Async retrieval (alias for retrieve).

        Args:
            query: Search query

        Returns:
            List of documents
        """
        return await self.retrieve(query)


def create_hybrid_retriever(
    bm25_k: int | None = None,
    dense_k: int | None = None,
    weights: List[float] | None = None,
    corpus_id: str | None = None
) -> PostgresHybridRetriever:
    """
    Create a hybrid retriever instance.

    Args:
        bm25_k: Number of BM25 results
        dense_k: Number of dense results
        weights: Weights for combining results
        corpus_id: Optional corpus filter

    Returns:
        PostgresHybridRetriever instance
    """
    return PostgresHybridRetriever(
        bm25_k=bm25_k,
        dense_k=dense_k,
        weights=weights,
        corpus_id=corpus_id
    )
