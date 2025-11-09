"""
RAPTOR retriever: Searches hierarchical summaries and expands to chunks.
"""
from typing import List, Optional
from uuid import UUID
from sqlalchemy import select
from langchain_core.documents import Document
from app.config import settings
from app.db.connection import get_db_session
from app.db.models import RAPTORSummary, RAPTORMembership, Chunk, Document as DocModel, Corpus
from app.retrieval.base import PostgresBaseRetriever, format_chunk_as_document
from app.ingestion.embedder import get_embedder
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAPTORRetriever(PostgresBaseRetriever):
    """
    RAPTOR retriever that searches hierarchical summaries.
    Can retrieve at specific levels or expand to underlying chunks.
    """

    k: int = 10  # Number of summaries to retrieve
    corpus_id: Optional[str] = None  # Required for RAPTOR (corpus-specific)
    level: Optional[int] = None  # Specific level to search (1, 2, 3, or None for all)
    expand_to_chunks: bool = True  # Whether to expand summaries to chunks

    def __init__(self, **kwargs):
        """Initialize RAPTOR retriever with embedder."""
        super().__init__(**kwargs)
        self.embedder = get_embedder()

        if not self.corpus_id:
            raise ValueError("corpus_id is required for RAPTOR retriever")

    async def _aget_relevant_documents_impl(
        self,
        query: str
    ) -> List[Document]:
        """
        Retrieve documents using RAPTOR summaries.

        Args:
            query: Search query

        Returns:
            List of relevant documents (summaries or expanded chunks)
        """
        logger.debug(
            "RAPTOR retrieval",
            query_preview=query[:100],
            k=self.k,
            corpus_id=self.corpus_id,
            level=self.level,
            expand_to_chunks=self.expand_to_chunks
        )

        try:
            # Generate query embedding
            query_embedding = await self.embedder.embed_text(query)

            async with get_db_session() as session:
                # Build query for summaries
                stmt = (
                    select(
                        RAPTORSummary.id,
                        RAPTORSummary.content,
                        RAPTORSummary.level,
                        RAPTORSummary.topic,
                        RAPTORSummary.metadata,
                        Corpus.name.label('corpus_name'),
                        RAPTORSummary.embedding.cosine_distance(query_embedding).label('distance')
                    )
                    .join(Corpus, RAPTORSummary.corpus_id == Corpus.id)
                    .where(RAPTORSummary.corpus_id == UUID(self.corpus_id))
                    .where(RAPTORSummary.embedding.isnot(None))
                )

                # Filter by level if specified
                if self.level is not None:
                    stmt = stmt.where(RAPTORSummary.level == self.level)

                # Order by similarity
                stmt = stmt.order_by('distance').limit(self.k)

                # Execute
                result = await session.execute(stmt)
                rows = result.all()

                if not self.expand_to_chunks:
                    # Return summaries directly
                    documents = []
                    for row in rows:
                        similarity = 1.0 - (float(row.distance) / 2.0)
                        doc = Document(
                            page_content=row.content,
                            metadata={
                                "id": str(row.id),
                                "type": "raptor_summary",
                                "level": row.level,
                                "topic": row.topic,
                                "corpus_name": row.corpus_name,
                                "similarity": similarity,
                                "retriever": "raptor",
                                **(row.metadata or {})
                            }
                        )
                        documents.append(doc)

                    logger.info(
                        "RAPTOR retrieval completed (summaries)",
                        results=len(documents),
                        levels=[d.metadata["level"] for d in documents]
                    )

                    return documents

                # Expand to chunks
                summary_ids = [row.id for row in rows]
                chunks = await self._expand_to_chunks(session, summary_ids)

                logger.info(
                    "RAPTOR retrieval completed (expanded to chunks)",
                    summaries=len(summary_ids),
                    chunks=len(chunks)
                )

                return chunks

        except Exception as e:
            logger.error(
                "RAPTOR retrieval failed",
                error=str(e),
                query=query[:100]
            )
            raise

    async def _expand_to_chunks(
        self,
        session,
        summary_ids: List[UUID]
    ) -> List[Document]:
        """
        Expand RAPTOR summaries to their underlying chunks.

        Args:
            session: Database session
            summary_ids: List of summary IDs

        Returns:
            List of chunk documents
        """
        # Get memberships
        result = await session.execute(
            select(
                Chunk.id,
                Chunk.content,
                Chunk.chunk_index,
                Chunk.section_type,
                Chunk.metadata,
                DocModel.filename,
                DocModel.title,
                Corpus.name.label('corpus_name'),
                RAPTORMembership.weight
            )
            .join(RAPTORMembership, Chunk.id == RAPTORMembership.chunk_id)
            .join(DocModel, Chunk.document_id == DocModel.id)
            .join(Corpus, Chunk.corpus_id == Corpus.id)
            .where(RAPTORMembership.summary_id.in_(summary_ids))
            .order_by(RAPTORMembership.weight.desc())
        )

        rows = result.all()

        # Convert to documents
        documents = []
        seen_ids = set()

        for row in rows:
            chunk_id = str(row.id)
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            doc = format_chunk_as_document(
                chunk_id=chunk_id,
                content=row.content,
                metadata={
                    "chunk_index": row.chunk_index,
                    "section_type": row.section_type,
                    "filename": row.filename,
                    "title": row.title,
                    "corpus_name": row.corpus_name,
                    "weight": float(row.weight),
                    "retriever": "raptor_expanded",
                    **(row.metadata or {})
                }
            )
            documents.append(doc)

        return documents


def create_raptor_retriever(
    corpus_id: str,
    k: int | None = None,
    level: int | None = None,
    expand_to_chunks: bool = True
) -> RAPTORRetriever:
    """
    Create a RAPTOR retriever instance.

    Args:
        corpus_id: Corpus UUID (required)
        k: Number of summaries to retrieve
        level: Specific level (1, 2, 3) or None for all
        expand_to_chunks: Whether to expand to chunks

    Returns:
        RAPTORRetriever instance
    """
    return RAPTORRetriever(
        k=k or 10,
        corpus_id=corpus_id,
        level=level,
        expand_to_chunks=expand_to_chunks
    )
