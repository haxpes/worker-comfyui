"""
Graph Builder: Construct knowledge graphs from entities and chunks.
Includes co-mention edges, community detection, and community summarization.
"""
import asyncio
from typing import List, Dict, Set, Optional, Tuple
from uuid import UUID
from collections import defaultdict
import networkx as nx
from community import community_louvain
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from app.config import settings
from app.db.connection import get_db_session
from app.db.models import (
    Entity, EntityOccurrence, Relationship,
    Community, CommunityMember, Chunk
)
from app.ingestion.embedder import get_embedder
from app.utils.logger import get_logger
from sqlalchemy import select, delete, func
import json

logger = get_logger(__name__)

COMMUNITY_SUMMARY_PROMPT = """You are an expert at analyzing entity clusters and generating concise topic summaries.

Entity Cluster:
{entities}

Task: Generate a concise 2-3 sentence summary describing:
1. The main topic/theme connecting these entities
2. The domain or context
3. Key relationships or patterns

Output only the summary, nothing else."""


class GraphBuilder:
    """
    Build knowledge graphs from extracted entities.
    Creates edges, detects communities, generates community summaries.
    """

    def __init__(self):
        """Initialize graph builder."""
        self.llm = ChatOpenAI(
            model=settings.default_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )
        self.embedder = get_embedder()

        logger.info("GraphBuilder initialized")

    async def build_graph(
        self,
        corpus_id: str,
        force_rebuild: bool = False,
        co_mention_window: int = 1,  # Same chunk
        min_edge_weight: float = 0.1,
        community_resolution: float = 1.0
    ) -> Dict[str, int]:
        """
        Build complete knowledge graph for a corpus.

        Args:
            corpus_id: Corpus ID
            force_rebuild: Delete existing graph first
            co_mention_window: Window size for co-mention (in chunks)
            min_edge_weight: Minimum edge weight to keep
            community_resolution: Louvain resolution parameter

        Returns:
            Statistics dictionary
        """
        logger.info(
            "Starting graph construction",
            corpus_id=corpus_id,
            force_rebuild=force_rebuild
        )

        async with get_db_session() as session:
            # Delete existing graph if force rebuild
            if force_rebuild:
                logger.info("Force rebuild: deleting existing graph")
                await session.execute(
                    delete(Relationship).where(Relationship.corpus_id == corpus_id)
                )
                await session.execute(
                    delete(Community).where(Community.corpus_id == corpus_id)
                )
                await session.commit()

            # Get all entities
            result = await session.execute(
                select(Entity).where(Entity.corpus_id == corpus_id)
            )
            entities = result.scalars().all()

            if len(entities) < 2:
                logger.warning(f"Not enough entities ({len(entities)}) for graph construction")
                return {"nodes": len(entities), "edges": 0, "communities": 0}

            logger.info(f"Building graph from {len(entities)} entities")

        # Build co-mention edges
        logger.info("Building co-mention edges...")
        edges = await self._build_co_mention_edges(
            corpus_id=corpus_id,
            window=co_mention_window
        )

        # Filter by minimum weight
        edges = [(s, t, w, e) for s, t, w, e in edges if w >= min_edge_weight]

        logger.info(f"Created {len(edges)} edges")

        # Store edges
        await self._store_relationships(corpus_id, edges)

        # Build NetworkX graph for community detection
        logger.info("Detecting communities...")
        G = nx.Graph()

        # Add nodes (entities)
        for entity in entities:
            G.add_node(str(entity.id), name=entity.name, type=entity.entity_type)

        # Add edges
        for source_id, target_id, weight, _ in edges:
            G.add_edge(str(source_id), str(target_id), weight=weight)

        # Detect communities using Louvain algorithm
        communities_dict = community_louvain.best_partition(
            G,
            weight='weight',
            resolution=community_resolution
        )

        # Group entities by community
        community_groups = defaultdict(list)
        for node_id, comm_id in communities_dict.items():
            community_groups[comm_id].append(node_id)

        logger.info(f"Detected {len(community_groups)} communities")

        # Create and store communities
        await self._create_communities(
            corpus_id=corpus_id,
            community_groups=community_groups,
            graph=G,
            entities={str(e.id): e for e in entities}
        )

        stats = {
            "nodes": len(entities),
            "edges": len(edges),
            "communities": len(community_groups)
        }

        logger.info("Graph construction completed", **stats)

        return stats

    async def _build_co_mention_edges(
        self,
        corpus_id: str,
        window: int
    ) -> List[Tuple[UUID, UUID, float, List[UUID]]]:
        """
        Build co-mention edges between entities that appear in same/nearby chunks.

        Args:
            corpus_id: Corpus ID
            window: Number of chunks for co-mention window

        Returns:
            List of (source_id, target_id, weight, evidence_chunks)
        """
        async with get_db_session() as session:
            # Get all entity occurrences
            result = await session.execute(
                select(EntityOccurrence, Entity)
                .join(Entity, EntityOccurrence.entity_id == Entity.id)
                .where(Entity.corpus_id == corpus_id)
                .order_by(EntityOccurrence.chunk_id)
            )

            occurrences = result.all()

        # Group by chunk
        chunk_entities = defaultdict(set)
        for occ, entity in occurrences:
            chunk_entities[occ.chunk_id].add(entity.id)

        # Build co-mention pairs
        edge_weights = defaultdict(lambda: {"weight": 0, "chunks": set()})

        for chunk_id, entity_ids in chunk_entities.items():
            entity_list = list(entity_ids)

            # All pairs in same chunk
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    e1, e2 = sorted([entity_list[i], entity_list[j]])
                    edge_key = (e1, e2)

                    edge_weights[edge_key]["weight"] += 1.0
                    edge_weights[edge_key]["chunks"].add(chunk_id)

        # Convert to edge list
        edges = []
        for (source_id, target_id), data in edge_weights.items():
            weight = data["weight"]
            evidence = list(data["chunks"])

            edges.append((source_id, target_id, weight, evidence))

        return edges

    async def _store_relationships(
        self,
        corpus_id: str,
        edges: List[Tuple[UUID, UUID, float, List[UUID]]]
    ):
        """
        Store relationships in database.

        Args:
            corpus_id: Corpus ID
            edges: List of edges
        """
        async with get_db_session() as session:
            for source_id, target_id, weight, evidence_chunks in edges:
                relationship = Relationship(
                    corpus_id=corpus_id,
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relationship_type="CO_MENTION",
                    weight=weight,
                    evidence_chunk_ids=evidence_chunks,
                    metadata={"evidence_count": len(evidence_chunks)}
                )
                session.add(relationship)

            await session.commit()

        logger.info(f"Stored {len(edges)} relationships")

    async def _create_communities(
        self,
        corpus_id: str,
        community_groups: Dict[int, List[str]],
        graph: nx.Graph,
        entities: Dict[str, Entity]
    ):
        """
        Create communities with summaries and embeddings.

        Args:
            corpus_id: Corpus ID
            community_groups: Community ID -> list of entity IDs
            graph: NetworkX graph
            entities: Entity ID -> Entity mapping
        """
        async with get_db_session() as session:
            for comm_id, entity_ids in community_groups.items():
                # Skip small communities
                if len(entity_ids) < 2:
                    continue

                # Get entity names and types
                entity_names = [
                    entities[eid].name for eid in entity_ids
                    if eid in entities
                ]

                # Calculate coherence score (average edge weight in community)
                subgraph = graph.subgraph(entity_ids)
                if subgraph.number_of_edges() > 0:
                    coherence = sum(
                        data.get('weight', 1.0)
                        for _, _, data in subgraph.edges(data=True)
                    ) / subgraph.number_of_edges()
                else:
                    coherence = 0.0

                # Generate summary
                summary = await self._summarize_community(entity_names)

                # Generate embedding
                summary_text = f"Community: {', '.join(entity_names[:10])}. {summary}"
                embedding = await self.embedder.embed(summary_text)

                # Create community
                community = Community(
                    corpus_id=corpus_id,
                    name=f"Community {comm_id}",
                    summary=summary,
                    embedding=embedding,
                    level=0,
                    size=len(entity_ids),
                    coherence_score=coherence,
                    metadata={"louvain_id": comm_id}
                )

                session.add(community)
                await session.flush()  # Get ID

                # Add members
                for entity_id_str in entity_ids:
                    if entity_id_str in entities:
                        member = CommunityMember(
                            community_id=community.id,
                            entity_id=entities[entity_id_str].id,
                            membership_score=1.0
                        )
                        session.add(member)

            await session.commit()

        logger.info(f"Created {len(community_groups)} communities")

    async def _summarize_community(self, entity_names: List[str]) -> str:
        """
        Generate LLM summary for a community of entities.

        Args:
            entity_names: List of entity names

        Returns:
            Community summary
        """
        try:
            # Format entities
            entities_text = "\n".join(f"- {name}" for name in entity_names[:20])

            prompt = ChatPromptTemplate.from_template(COMMUNITY_SUMMARY_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            summary = await chain.ainvoke({"entities": entities_text})

            return summary.strip()

        except Exception as e:
            logger.warning("Community summary failed", error=str(e))
            return f"Community of {len(entity_names)} entities including {', '.join(entity_names[:3])}"


def create_graph_builder() -> GraphBuilder:
    """
    Create a GraphBuilder instance.

    Returns:
        GraphBuilder
    """
    return GraphBuilder()
