"""
Dense vector retriever using pgvector.
Uses cosine similarity for semantic search.
"""
from typing import List, Optional
from uuid import UUID
from sqlalchemy import select
from langchain_core.documents import Document
from app.config import settings
from app.db.connection import get_db_session
from app.db.models import Chunk, Document as DocModel, Corpus
from app.retrieval.base import PostgresBaseRetriever, format_chunk_as_document
from app.ingestion.embedder import get_embedder
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PostgresDenseRetriever(PostgresBaseRetriever):
    """
    Dense vector retriever using pgvector for semantic search.
    """

    k: int = 20  # Default top-k
    corpus_id: Optional[str] = None  # Optional corpus filter

    def __init__(self, **kwargs):
        """Initialize dense retriever with embedder."""
        super().__init__(**kwargs)
        self.embedder = get_embedder()

    async def _aget_relevant_documents_impl(
        self,
        query: str
    ) -> List[Document]:
        """
        Retrieve documents using vector similarity search.

        Args:
            query: Search query

        Returns:
            List of relevant documents sorted by similarity
        """
        logger.debug(
            "Dense retrieval",
            query_preview=query[:100],
            k=self.k,
            corpus_id=self.corpus_id
        )

        try:
            # Generate query embedding
            query_embedding = await self.embedder.embed_text(query)

            async with get_db_session() as session:
                # Build similarity query using pgvector's <=> operator (cosine distance)
                # Note: pgvector uses distance (lower is better), we convert to similarity

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
                        # Cosine distance (0 = identical, 2 = opposite)
                        Chunk.embedding.cosine_distance(query_embedding).label('distance')
                    )
                    .join(DocModel, Chunk.document_id == DocModel.id)
                    .join(Corpus, Chunk.corpus_id == Corpus.id)
                    .where(Chunk.embedding.isnot(None))  # Only chunks with embeddings
                )

                # Apply corpus filter if specified
                if self.corpus_id:
                    stmt = stmt.where(Chunk.corpus_id == UUID(self.corpus_id))

                # Order by distance (ascending) and limit
                stmt = stmt.order_by('distance').limit(self.k)

                # Execute query
                result = await session.execute(stmt)
                rows = result.all()

                # Convert to Documents
                documents = []
                for row in rows:
                    # Convert distance to similarity score (0-1, where 1 is most similar)
                    similarity = 1.0 - (float(row.distance) / 2.0)

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
                            "similarity": similarity,
                            "distance": float(row.distance),
                            "retriever": "dense",
                            **(row.metadata or {})
                        }
                    )
                    documents.append(doc)

                logger.info(
                    "Dense retrieval completed",
                    query_preview=query[:50],
                    results=len(documents),
                    k=self.k,
                    avg_similarity=round(sum(d.metadata["similarity"] for d in documents) / len(documents), 3) if documents else 0
                )

                return documents

        except Exception as e:
            logger.error(
                "Dense retrieval failed",
                error=str(e),
                query=query[:100]
            )
            raise


def create_dense_retriever(
    k: int | None = None,
    corpus_id: str | None = None
) -> PostgresDenseRetriever:
    """
    Create a dense vector retriever instance.

    Args:
        k: Number of results (defaults to config)
        corpus_id: Optional corpus filter

    Returns:
        PostgresDenseRetriever instance
    """
    return PostgresDenseRetriever(
        k=k or settings.dense_k,
        corpus_id=corpus_id
    )
