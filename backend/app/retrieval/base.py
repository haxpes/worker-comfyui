"""
Base retriever interface and utilities.
All retrievers inherit from LangChain's BaseRetriever.
"""
from typing import List, Optional
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PostgresBaseRetriever(BaseRetriever, ABC):
    """
    Base class for PostgreSQL-backed retrievers.
    Inherits from LangChain's BaseRetriever for compatibility.
    """

    k: int = 10  # Number of results to return
    corpus_id: Optional[str] = None  # Optional corpus filter

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True

    @abstractmethod
    async def _aget_relevant_documents_impl(
        self,
        query: str
    ) -> List[Document]:
        """
        Implementation of retrieval logic (to be overridden).

        Args:
            query: Search query

        Returns:
            List of relevant documents
        """
        pass

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        """
        Synchronous retrieval (required by BaseRetriever).
        Not used in async context but required by interface.
        """
        import asyncio
        return asyncio.run(self._aget_relevant_documents_impl(query))

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        """
        Asynchronous retrieval (main method).

        Args:
            query: Search query
            run_manager: Callback manager (optional)

        Returns:
            List of relevant documents
        """
        return await self._aget_relevant_documents_impl(query)


def format_chunk_as_document(
    chunk_id: str,
    content: str,
    metadata: dict
) -> Document:
    """
    Convert a chunk to LangChain Document format.

    Args:
        chunk_id: Chunk UUID
        content: Chunk content
        metadata: Additional metadata

    Returns:
        LangChain Document
    """
    return Document(
        page_content=content,
        metadata={
            "id": chunk_id,
            **metadata
        }
    )


def deduplicate_documents(
    documents: List[Document],
    key: str = "id"
) -> List[Document]:
    """
    Remove duplicate documents based on a metadata key.

    Args:
        documents: List of documents
        key: Metadata key to use for deduplication

    Returns:
        Deduplicated list of documents
    """
    seen = set()
    unique_docs = []

    for doc in documents:
        doc_id = doc.metadata.get(key)
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            unique_docs.append(doc)
        elif not doc_id:
            # No ID, keep it
            unique_docs.append(doc)

    logger.debug(
        "Deduplicated documents",
        original_count=len(documents),
        unique_count=len(unique_docs),
        duplicates_removed=len(documents) - len(unique_docs)
    )

    return unique_docs


def merge_document_lists(
    *doc_lists: List[Document],
    deduplicate: bool = True
) -> List[Document]:
    """
    Merge multiple document lists.

    Args:
        *doc_lists: Variable number of document lists
        deduplicate: Whether to remove duplicates

    Returns:
        Merged list of documents
    """
    merged = []
    for doc_list in doc_lists:
        merged.extend(doc_list)

    if deduplicate:
        merged = deduplicate_documents(merged)

    return merged


def truncate_documents(
    documents: List[Document],
    max_docs: int
) -> List[Document]:
    """
    Truncate document list to maximum size.

    Args:
        documents: List of documents
        max_docs: Maximum number of documents

    Returns:
        Truncated list
    """
    if len(documents) <= max_docs:
        return documents

    logger.debug(
        "Truncating documents",
        original_count=len(documents),
        max_docs=max_docs
    )

    return documents[:max_docs]
