"""
BM25 retriever using PostgreSQL full-text search.
Uses ts_rank for ranking based on tsvector columns.
"""
from typing import List, Optional
from uuid import UUID
from sqlalchemy import select, func, text
from langchain_core.documents import Document
from app.config import settings
from app.db.connection import get_db_session
from app.db.models import Chunk, Document as DocModel, Corpus
from app.retrieval.base import PostgresBaseRetriever, format_chunk_as_document
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PostgresBM25Retriever(PostgresBaseRetriever):
    """
    BM25-style full-text retriever using PostgreSQL.
    Uses ts_rank_cd for ranking (similar to BM25).
    """

    k: int = 20  # Default top-k
    corpus_id: Optional[str] = None  # Optional corpus filter

    async def _aget_relevant_documents_impl(
        self,
        query: str
    ) -> List[Document]:
        """
        Retrieve documents using full-text search.

        Args:
            query: Search query

        Returns:
            List of relevant documents sorted by relevance
        """
        logger.debug(
            "BM25 retrieval",
            query_preview=query[:100],
            k=self.k,
            corpus_id=self.corpus_id
        )

        try:
            async with get_db_session() as session:
                # Build full-text query
                # Use plainto_tsquery for simple query parsing
                ts_query = func.plainto_tsquery('english', query)

                # Base query
                stmt = (
                    select(
                        Chunk.id,
                        Chunk.content,
                        Chunk.chunk_index,
                        Chunk.section_type,
                        Chunk.metadata,
                        Chunk.document_id,
                        Chunk.corpus_id,
                        DocModel.filename,
                        DocModel.title,
                        Corpus.name.label('corpus_name'),
                        # Ranking function (ts_rank_cd for better BM25 approximation)
                        func.ts_rank_cd(Chunk.content_tsv, ts_query).label('rank')
                    )
                    .join(DocModel, Chunk.document_id == DocModel.id)
                    .join(Corpus, Chunk.corpus_id == Corpus.id)
                    .where(Chunk.content_tsv.op('@@')(ts_query))
                )

                # Apply corpus filter if specified
                if self.corpus_id:
                    stmt = stmt.where(Chunk.corpus_id == UUID(self.corpus_id))

                # Order by rank and limit
                stmt = stmt.order_by(text('rank DESC')).limit(self.k)

                # Execute query
                result = await session.execute(stmt)
                rows = result.all()

                # Convert to Documents
                documents = []
                for row in rows:
                    doc = format_chunk_as_document(
                        chunk_id=str(row.id),
                        content=row.content,
                        metadata={
                            "chunk_index": row.chunk_index,
                            "section_type": row.section_type,
                            "document_id": str(row.document_id),
                            "corpus_id": str(row.corpus_id),
                            "filename": row.filename,
                            "title": row.title,
                            "corpus_name": row.corpus_name,
                            "rank": float(row.rank),
                            "retriever": "bm25",
                            **(row.metadata or {})
                        }
                    )
                    documents.append(doc)

                logger.info(
                    "BM25 retrieval completed",
                    query_preview=query[:50],
                    results=len(documents),
                    k=self.k
                )

                return documents

        except Exception as e:
            logger.error(
                "BM25 retrieval failed",
                error=str(e),
                query=query[:100]
            )
            raise


def create_bm25_retriever(
    k: int | None = None,
    corpus_id: str | None = None
) -> PostgresBM25Retriever:
    """
    Create a BM25 retriever instance.

    Args:
        k: Number of results (defaults to config)
        corpus_id: Optional corpus filter

    Returns:
        PostgresBM25Retriever instance
    """
    return PostgresBM25Retriever(
        k=k or settings.bm25_k,
        corpus_id=corpus_id
    )
