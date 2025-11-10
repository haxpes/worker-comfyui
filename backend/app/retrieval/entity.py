"""
Entity Retriever: Search entities and expand to related chunks.
"""
from typing import List, Optional, Dict, Any
from uuid import UUID
from langchain_core.documents import Document
from app.config import settings
from app.db.connection import get_db_session
from app.db.models import Entity, EntityOccurrence, Chunk
from app.retrieval.base import PostgresBaseRetriever
from app.utils.logger import get_logger
from sqlalchemy import select, or_, func
from sqlalchemy.orm import joinedload

logger = get_logger(__name__)


class EntityRetriever(PostgresBaseRetriever):
    """
    Retrieve entities by name/type and expand to associated chunks.
    """

    def __init__(
        self,
        corpus_id: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
        expand_to_chunks: bool = True,
        k: int = 10
    ):
        """
        Initialize entity retriever.

        Args:
            corpus_id: Optional corpus ID filter
            entity_types: Optional entity type filter
            expand_to_chunks: Whether to expand entities to their chunks
            k: Number of entities to retrieve
        """
        super().__init__()
        self.corpus_id = corpus_id
        self.entity_types = entity_types or []
        self.expand_to_chunks = expand_to_chunks
        self.k = k

        logger.info(
            "EntityRetriever initialized",
            corpus_id=corpus_id,
            types=entity_types,
            k=k
        )

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve entities matching query.

        Args:
            query: Search query (entity name keywords)

        Returns:
            List of documents (either entities or expanded chunks)
        """
        logger.info("Retrieving entities", query_preview=query[:100])

        async with get_db_session() as session:
            # Build entity search query
            stmt = select(Entity).options(
                joinedload(Entity.occurrences).joinedload(EntityOccurrence.chunk)
            )

            # Corpus filter
            if self.corpus_id:
                stmt = stmt.where(Entity.corpus_id == self.corpus_id)

            # Type filter
            if self.entity_types:
                stmt = stmt.where(Entity.entity_type.in_(self.entity_types))

            # Name search (case-insensitive partial match)
            query_terms = query.lower().split()
            for term in query_terms:
                stmt = stmt.where(Entity.name.ilike(f"%{term}%"))

            # Order by confidence, limit
            stmt = stmt.order_by(Entity.confidence.desc()).limit(self.k)

            result = await session.execute(stmt)
            entities = result.unique().scalars().all()

            logger.info(f"Found {len(entities)} matching entities")

            # Convert to documents
            if self.expand_to_chunks:
                # Expand to chunks where entities appear
                documents = []
                seen_chunks = set()

                for entity in entities:
                    for occurrence in entity.occurrences:
                        chunk = occurrence.chunk

                        if chunk.id in seen_chunks:
                            continue

                        seen_chunks.add(chunk.id)

                        doc = Document(
                            page_content=chunk.content,
                            metadata={
                                "chunk_id": str(chunk.id),
                                "doc_id": str(chunk.doc_id),
                                "entity_name": entity.name,
                                "entity_type": entity.entity_type,
                                "entity_id": str(entity.id),
                                "entity_context": occurrence.context
                            }
                        )
                        documents.append(doc)

                logger.info(f"Expanded to {len(documents)} chunks")
                return documents[:self.k]

            else:
                # Return entities directly
                documents = [
                    Document(
                        page_content=f"{entity.name} ({entity.entity_type}): {entity.description or 'No description'}",
                        metadata={
                            "entity_id": str(entity.id),
                            "name": entity.name,
                            "type": entity.entity_type,
                            "confidence": entity.confidence,
                            "occurrence_count": len(entity.occurrences)
                        }
                    )
                    for entity in entities
                ]

                return documents


def create_entity_retriever(
    corpus_id: Optional[str] = None,
    entity_types: Optional[List[str]] = None,
    **kwargs
) -> EntityRetriever:
    """
    Create an EntityRetriever instance.

    Args:
        corpus_id: Optional corpus ID
        entity_types: Optional entity type filter
        **kwargs: Additional arguments

    Returns:
        EntityRetriever
    """
    return EntityRetriever(
        corpus_id=corpus_id,
        entity_types=entity_types,
        **kwargs
    )
