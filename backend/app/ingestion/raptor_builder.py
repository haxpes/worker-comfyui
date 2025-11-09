"""
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) builder.
Creates hierarchical summaries of document chunks.

Algorithm:
1. L0: Original chunks
2. L1: Cluster chunks → create summaries
3. L2: Cluster L1 summaries → create meta-summaries
4. L3: Corpus-level summary

Uses clustering + LLM summarization to create a tree structure.
"""
import asyncio
from typing import List, Dict, Any, Tuple
from uuid import UUID
import numpy as np
from sklearn.cluster import KMeans
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import select
from app.config import settings
from app.db.connection import get_db_session
from app.db.models import Chunk, RAPTORSummary, RAPTORMembership
from app.ingestion.embedder import get_embedder
from app.utils.logger import get_logger

logger = get_logger(__name__)


# Summarization prompts
CLUSTER_SUMMARY_PROMPT = """You are an expert at creating concise, informative summaries.

Given the following related text chunks, create a comprehensive summary that captures the main themes, key points, and important details.

Chunks:
{chunks}

Create a clear, well-structured summary (2-4 paragraphs) that:
1. Identifies the main topic or theme
2. Captures key points from all chunks
3. Maintains important details
4. Is coherent and well-organized

Summary:"""

CORPUS_SUMMARY_PROMPT = """You are an expert at creating executive summaries.

Given the following high-level summaries, create a comprehensive corpus-level overview.

Summaries:
{summaries}

Create a well-structured overview (3-5 paragraphs) that:
1. Identifies major themes and topics
2. Highlights key concepts and relationships
3. Provides a coherent narrative
4. Captures the essence of the entire corpus

Overview:"""


class RAPTORBuilder:
    """
    Builds RAPTOR hierarchical summaries for a corpus.
    """

    def __init__(self):
        """Initialize RAPTOR builder."""
        self.embedder = get_embedder()
        self.llm = ChatOpenAI(
            model=settings.complex_model,  # Use GPT-4 for better summarization
            temperature=0,
            openai_api_key=settings.openai_api_key
        )

        # Clustering parameters
        self.l1_max_clusters = 50  # Max clusters for L1
        self.l2_max_clusters = 10  # Max clusters for L2
        self.min_cluster_size = 3  # Min chunks per cluster

        logger.info("RAPTOR builder initialized")

    async def build_raptor_tree(
        self,
        corpus_id: UUID,
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        Build complete RAPTOR tree for a corpus.

        Args:
            corpus_id: Corpus UUID
            force_rebuild: If True, delete existing summaries

        Returns:
            Statistics dictionary
        """
        logger.info(
            "Building RAPTOR tree",
            corpus_id=str(corpus_id),
            force_rebuild=force_rebuild
        )

        # Clean existing summaries if rebuilding
        if force_rebuild:
            await self._clean_existing_summaries(corpus_id)

        # Load chunks
        chunks = await self._load_chunks(corpus_id)

        if not chunks:
            logger.warning("No chunks found for corpus", corpus_id=str(corpus_id))
            return {
                "corpus_id": str(corpus_id),
                "l1_summaries": 0,
                "l2_summaries": 0,
                "l3_summaries": 0
            }

        logger.info(f"Loaded {len(chunks)} chunks")

        # Build L1: Cluster chunks → summaries
        l1_summaries = await self._build_l1_summaries(corpus_id, chunks)
        logger.info(f"Created {len(l1_summaries)} L1 summaries")

        # Build L2: Cluster L1 summaries → meta-summaries
        l2_summaries = []
        if len(l1_summaries) > 1:
            l2_summaries = await self._build_l2_summaries(corpus_id, l1_summaries)
            logger.info(f"Created {len(l2_summaries)} L2 summaries")

        # Build L3: Corpus-level summary
        l3_summary = None
        if len(l2_summaries) > 0:
            l3_summary = await self._build_l3_summary(corpus_id, l2_summaries)
            logger.info("Created L3 corpus summary")
        elif len(l1_summaries) > 0:
            # Fallback: use L1 summaries if no L2
            l3_summary = await self._build_l3_summary(corpus_id, l1_summaries)
            logger.info("Created L3 corpus summary (from L1)")

        stats = {
            "corpus_id": str(corpus_id),
            "chunks": len(chunks),
            "l1_summaries": len(l1_summaries),
            "l2_summaries": len(l2_summaries),
            "l3_summaries": 1 if l3_summary else 0
        }

        logger.info("RAPTOR tree built successfully", **stats)

        return stats

    async def _load_chunks(self, corpus_id: UUID) -> List[Dict[str, Any]]:
        """Load all chunks for a corpus with embeddings."""
        async with get_db_session() as session:
            result = await session.execute(
                select(Chunk)
                .where(Chunk.corpus_id == corpus_id)
                .where(Chunk.embedding.isnot(None))
                .order_by(Chunk.chunk_index)
            )
            chunks = result.scalars().all()

            return [
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "embedding": chunk.embedding,
                    "metadata": chunk.metadata
                }
                for chunk in chunks
            ]

    async def _clean_existing_summaries(self, corpus_id: UUID):
        """Delete existing RAPTOR summaries for corpus."""
        async with get_db_session() as session:
            result = await session.execute(
                select(RAPTORSummary).where(RAPTORSummary.corpus_id == corpus_id)
            )
            summaries = result.scalars().all()

            for summary in summaries:
                await session.delete(summary)

            await session.commit()

            logger.info(f"Deleted {len(summaries)} existing summaries")

    async def _build_l1_summaries(
        self,
        corpus_id: UUID,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build L1 summaries by clustering chunks."""

        # Extract embeddings
        embeddings = np.array([chunk["embedding"] for chunk in chunks])

        # Determine number of clusters
        n_clusters = min(self.l1_max_clusters, len(chunks) // self.min_cluster_size)
        n_clusters = max(1, n_clusters)

        logger.info(f"Clustering {len(chunks)} chunks into {n_clusters} clusters")

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Group chunks by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(chunks[i])

        # Create summaries for each cluster
        l1_summaries = []

        for cluster_id, cluster_chunks in clusters.items():
            if len(cluster_chunks) < self.min_cluster_size:
                logger.debug(f"Skipping small cluster {cluster_id} ({len(cluster_chunks)} chunks)")
                continue

            summary_data = await self._summarize_cluster(
                cluster_chunks,
                level=1,
                cluster_id=cluster_id
            )

            if summary_data:
                # Store in database
                summary_id = await self._store_summary(
                    corpus_id=corpus_id,
                    level=1,
                    content=summary_data["content"],
                    embedding=summary_data["embedding"],
                    topic=summary_data["topic"],
                    parent_id=None
                )

                # Store membership relationships
                await self._store_memberships(
                    summary_id=summary_id,
                    chunk_ids=[chunk["id"] for chunk in cluster_chunks]
                )

                l1_summaries.append({
                    "id": summary_id,
                    "content": summary_data["content"],
                    "embedding": summary_data["embedding"],
                    "topic": summary_data["topic"]
                })

        return l1_summaries

    async def _build_l2_summaries(
        self,
        corpus_id: UUID,
        l1_summaries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build L2 summaries by clustering L1 summaries."""

        if len(l1_summaries) < self.min_cluster_size:
            logger.info("Not enough L1 summaries for L2 clustering")
            return []

        # Extract embeddings
        embeddings = np.array([summary["embedding"] for summary in l1_summaries])

        # Determine number of clusters
        n_clusters = min(self.l2_max_clusters, len(l1_summaries) // self.min_cluster_size)
        n_clusters = max(1, n_clusters)

        logger.info(f"Clustering {len(l1_summaries)} L1 summaries into {n_clusters} clusters")

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Group summaries by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(l1_summaries[i])

        # Create L2 summaries
        l2_summaries = []

        for cluster_id, cluster_summaries in clusters.items():
            if len(cluster_summaries) < 2:
                logger.debug(f"Skipping small L2 cluster {cluster_id}")
                continue

            # Treat L1 summaries as "chunks" for summarization
            summary_data = await self._summarize_cluster(
                [{"id": s["id"], "content": s["content"], "metadata": {}}
                 for s in cluster_summaries],
                level=2,
                cluster_id=cluster_id
            )

            if summary_data:
                # Store in database
                summary_id = await self._store_summary(
                    corpus_id=corpus_id,
                    level=2,
                    content=summary_data["content"],
                    embedding=summary_data["embedding"],
                    topic=summary_data["topic"],
                    parent_id=None
                )

                # Link L1 summaries as children
                # Note: We don't use parent_id here, we use membership for chunks only

                l2_summaries.append({
                    "id": summary_id,
                    "content": summary_data["content"],
                    "embedding": summary_data["embedding"],
                    "topic": summary_data["topic"]
                })

        return l2_summaries

    async def _build_l3_summary(
        self,
        corpus_id: UUID,
        summaries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build L3 corpus-level summary."""

        # Combine all summaries
        combined_text = "\n\n---\n\n".join([s["content"] for s in summaries])

        # Create corpus-level summary
        prompt = ChatPromptTemplate.from_template(CORPUS_SUMMARY_PROMPT)
        chain = prompt | self.llm | StrOutputParser()

        logger.info("Generating L3 corpus summary")

        summary_text = await chain.ainvoke({"summaries": combined_text})

        # Generate embedding
        embedding = await self.embedder.embed_text(summary_text)

        # Extract topic (first sentence or first 100 chars)
        topic = summary_text.split(".")[0][:100]

        # Store in database
        summary_id = await self._store_summary(
            corpus_id=corpus_id,
            level=3,
            content=summary_text,
            embedding=embedding,
            topic=topic,
            parent_id=None
        )

        return {
            "id": summary_id,
            "content": summary_text,
            "embedding": embedding,
            "topic": topic
        }

    async def _summarize_cluster(
        self,
        cluster_chunks: List[Dict[str, Any]],
        level: int,
        cluster_id: int
    ) -> Dict[str, Any] | None:
        """Summarize a cluster of chunks/summaries."""

        # Combine chunks
        combined_text = "\n\n---\n\n".join([chunk["content"] for chunk in cluster_chunks])

        # Truncate if too long (GPT-4 context limit)
        max_chars = 10000
        if len(combined_text) > max_chars:
            combined_text = combined_text[:max_chars] + "\n\n[...truncated...]"

        # Generate summary
        prompt = ChatPromptTemplate.from_template(CLUSTER_SUMMARY_PROMPT)
        chain = prompt | self.llm | StrOutputParser()

        try:
            summary_text = await chain.ainvoke({"chunks": combined_text})

            # Generate embedding
            embedding = await self.embedder.embed_text(summary_text)

            # Extract topic (first sentence)
            topic = summary_text.split(".")[0][:100]

            return {
                "content": summary_text,
                "embedding": embedding,
                "topic": topic
            }

        except Exception as e:
            logger.error(
                "Failed to summarize cluster",
                level=level,
                cluster_id=cluster_id,
                error=str(e)
            )
            return None

    async def _store_summary(
        self,
        corpus_id: UUID,
        level: int,
        content: str,
        embedding: List[float],
        topic: str,
        parent_id: UUID | None
    ) -> UUID:
        """Store RAPTOR summary in database."""
        async with get_db_session() as session:
            summary = RAPTORSummary(
                corpus_id=corpus_id,
                level=level,
                content=content,
                embedding=embedding,
                topic=topic,
                parent_id=parent_id
            )

            session.add(summary)
            await session.commit()
            await session.refresh(summary)

            return summary.id

    async def _store_memberships(
        self,
        summary_id: UUID,
        chunk_ids: List[UUID]
    ):
        """Store RAPTOR membership relationships."""
        async with get_db_session() as session:
            memberships = [
                RAPTORMembership(
                    summary_id=summary_id,
                    chunk_id=chunk_id,
                    weight=1.0
                )
                for chunk_id in chunk_ids
            ]

            session.add_all(memberships)
            await session.commit()


async def build_raptor_tree(
    corpus_id: UUID,
    force_rebuild: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to build RAPTOR tree.

    Args:
        corpus_id: Corpus UUID
        force_rebuild: Whether to rebuild from scratch

    Returns:
        Statistics dictionary
    """
    builder = RAPTORBuilder()
    return await builder.build_raptor_tree(corpus_id, force_rebuild)
