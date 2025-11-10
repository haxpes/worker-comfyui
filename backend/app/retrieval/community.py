"""
Community Retriever: Search entity communities and expand to member chunks.
"""
from typing import List, Optional
from langchain_core.documents import Document
from app.db.connection import get_db_session
from app.db.models import Community, CommunityMember, Entity, EntityOccurrence
from app.ingestion.embedder import get_embedder
from app.retrieval.base import PostgresBaseRetriever
from app.utils.logger import get_logger
from sqlalchemy import select, func
from sqlalchemy.orm import joinedload

logger = get_logger(__name__)


class CommunityRetriever(PostgresBaseRetriever):
    """
    Retrieve entity communities by semantic search and expand to chunks.
    """

    def __init__(
        self,
        corpus_id: Optional[str] = None,
        expand_to_chunks: bool = True,
        k: int = 5,
        max_entities_per_community: int = 20
    ):
        """
        Initialize community retriever.

        Args:
            corpus_id: Optional corpus ID filter
            expand_to_chunks: Whether to expand to entity chunks
            k: Number of communities to retrieve
            max_entities_per_community: Max entities to include per community
        """
        super().__init__()
        self.corpus_id = corpus_id
        self.expand_to_chunks = expand_to_chunks
        self.k = k
        self.max_entities_per_community = max_entities_per_community
        self.embedder = get_embedder()

        logger.info(
            "CommunityRetriever initialized",
            corpus_id=corpus_id,
            k=k
        )

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve communities matching query.

        Args:
            query: Search query

        Returns:
            List of documents (community summaries or expanded chunks)
        """
        logger.info("Retrieving communities", query_preview=query[:100])

        # Embed query
        query_embedding = await self.embedder.embed(query)

        async with get_db_session() as session:
            # Build community search query with cosine similarity
            stmt = (
                select(
                    Community,
                    Community.embedding.cosine_distance(query_embedding).label('distance')
                )
                .options(
                    joinedload(Community.members).joinedload(CommunityMember.entity)
                )
            )

            # Corpus filter
            if self.corpus_id:
                stmt = stmt.where(Community.corpus_id == self.corpus_id)

            # Filter out communities without embeddings
            stmt = stmt.where(Community.embedding.isnot(None))

            # Order by similarity, limit
            stmt = stmt.order_by('distance').limit(self.k)

            result = await session.execute(stmt)
            rows = result.all()

            communities = [row[0] for row in rows]
            distances = [row[1] for row in rows]

            logger.info(f"Found {len(communities)} matching communities")

            # Convert to documents
            if self.expand_to_chunks:
                # Expand to chunks from community entities
                documents = []

                for community in communities:
                    # Get entity IDs in community
                    entity_ids = [
                        member.entity_id
                        for member in community.members[:self.max_entities_per_community]
                    ]

                    # Get entity occurrences
                    occ_result = await session.execute(
                        select(EntityOccurrence, Entity)
                        .join(Entity, EntityOccurrence.entity_id == Entity.id)
                        .where(EntityOccurrence.entity_id.in_(entity_ids))
                        .options(joinedload(EntityOccurrence.chunk))
                        .limit(50)  # Limit total chunks per community
                    )

                    occurrences = occ_result.all()

                    # Create documents from chunks
                    seen_chunks = set()
                    for occ, entity in occurrences:
                        chunk = occ.chunk

                        if chunk.id in seen_chunks:
                            continue

                        seen_chunks.add(chunk.id)

                        doc = Document(
                            page_content=chunk.content,
                            metadata={
                                "chunk_id": str(chunk.id),
                                "doc_id": str(chunk.doc_id),
                                "community_id": str(community.id),
                                "community_name": community.name,
                                "community_summary": community.summary,
                                "entity_name": entity.name,
                                "entity_type": entity.entity_type
                            }
                        )
                        documents.append(doc)

                logger.info(f"Expanded to {len(documents)} chunks from communities")
                return documents

            else:
                # Return community summaries directly
                documents = [
                    Document(
                        page_content=f"{community.name}: {community.summary or 'No summary'}",
                        metadata={
                            "community_id": str(community.id),
                            "name": community.name,
                            "size": community.size,
                            "coherence": community.coherence_score,
                            "level": community.level,
                            "similarity": 1.0 - dist  # Convert distance to similarity
                        }
                    )
                    for community, dist in zip(communities, distances)
                ]

                return documents


def create_community_retriever(
    corpus_id: Optional[str] = None,
    **kwargs
) -> CommunityRetriever:
    """
    Create a CommunityRetriever instance.

    Args:
        corpus_id: Optional corpus ID
        **kwargs: Additional arguments

    Returns:
        CommunityRetriever
    """
    return CommunityRetriever(corpus_id=corpus_id, **kwargs)
