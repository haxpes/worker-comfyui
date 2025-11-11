"""
FAISS Index Builder for Fast Similarity Search.

Provides O(log n) similarity search for deduplication, relevance pre-filtering,
and diversity calculation.
"""
from typing import List, Tuple, Dict, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class FAISSIndexBuilder:
    """
    FAISS index builder for fast similarity search.

    Provides:
    - Fast similarity search (O(log n))
    - Scalable to thousands of sentences
    - In-memory with optional persistence
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize FAISS index builder.

        Args:
            embedding_model: Sentence transformer model name
        """
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index: Optional[faiss.Index] = None
        self.sentence_map: Dict[int, str] = {}
        self.embeddings: Optional[np.ndarray] = None

        logger.info(
            "FAISS index builder initialized",
            model=embedding_model
        )

    def build_index(self, sentences: List[str]) -> faiss.Index:
        """
        Build FAISS index from sentences.

        Args:
            sentences: List of sentences to index

        Returns:
            FAISS index
        """
        logger.info("Building FAISS index", sentences=len(sentences))

        # Generate embeddings
        self.embeddings = self.embedding_model.encode(
            sentences,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )

        # Dimension
        dimension = self.embeddings.shape[1]

        # Choose index type based on size
        if len(sentences) < 10000:
            # Simple index for small sets
            self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine
            logger.debug("Using IndexFlatIP for small dataset")

        else:
            # Clustered index for large sets
            nlist = min(100, len(sentences) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

            # Train index
            if len(sentences) >= nlist:
                logger.debug(f"Training IVF index with {nlist} clusters")
                self.index.train(self.embeddings)

            self.index.nprobe = 10  # Search in 10 nearest clusters
            logger.debug("Using IndexIVFFlat for large dataset")

        # Add embeddings
        self.index.add(self.embeddings)

        # Map FAISS IDs to sentences
        for i, sentence in enumerate(sentences):
            self.sentence_map[i] = sentence

        logger.info(
            "FAISS index built",
            sentences=len(sentences),
            dimension=dimension
        )

        return self.index

    def search(
        self,
        query: str | np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar sentences.

        Args:
            query: Query string or embedding
            k: Number of results

        Returns:
            (distances, indices)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Get query embedding
        if isinstance(query, str):
            query_embedding = self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        else:
            query_embedding = query.reshape(1, -1)
            # Normalize
            faiss.normalize_L2(query_embedding)

        # Search
        distances, indices = self.index.search(query_embedding, k)

        return distances, indices

    def get_sentence(self, idx: int) -> Optional[str]:
        """
        Get sentence by FAISS index.

        Args:
            idx: FAISS index

        Returns:
            Sentence or None
        """
        return self.sentence_map.get(idx)

    def get_sentences(self, indices: List[int]) -> List[str]:
        """
        Get multiple sentences by indices.

        Args:
            indices: List of FAISS indices

        Returns:
            List of sentences
        """
        return [
            self.sentence_map.get(idx, "")
            for idx in indices
        ]

    def find_similar_sentences(
        self,
        sentence: str,
        k: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Find similar sentences with similarity threshold.

        Args:
            sentence: Query sentence
            k: Number of results
            similarity_threshold: Minimum similarity

        Returns:
            List of (sentence, similarity) tuples
        """
        distances, indices = self.search(sentence, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            # Convert distance to similarity (for normalized vectors)
            similarity = float(dist)  # Already cosine similarity for IndexFlatIP

            if similarity >= similarity_threshold:
                sent = self.sentence_map.get(idx)
                if sent:
                    results.append((sent, similarity))

        return results

    def deduplicate_sentences(
        self,
        sentences: List[str],
        similarity_threshold: float = 0.85
    ) -> List[str]:
        """
        Deduplicate sentences using FAISS.

        Args:
            sentences: List of sentences
            similarity_threshold: Similarity threshold for duplicates

        Returns:
            Deduplicated sentences
        """
        logger.info(
            "Deduplicating sentences",
            sentences=len(sentences),
            threshold=similarity_threshold
        )

        unique_sentences = []
        seen_indices = set()

        for i, sentence in enumerate(sentences):
            if i in seen_indices:
                continue

            # Find similar sentences
            similar = self.find_similar_sentences(
                sentence,
                k=len(sentences),
                similarity_threshold=similarity_threshold
            )

            # Mark duplicates
            for similar_sent, similarity in similar:
                similar_idx = next(
                    (idx for idx, s in self.sentence_map.items() if s == similar_sent),
                    -1
                )
                if similar_idx != -1 and similar_idx != i:
                    seen_indices.add(similar_idx)

            # Keep sentence
            unique_sentences.append(sentence)

        logger.info(
            "Deduplication completed",
            original=len(sentences),
            unique=len(unique_sentences),
            removed=len(sentences) - len(unique_sentences)
        )

        return unique_sentences

    def get_embedding(self, sentence: str) -> np.ndarray:
        """
        Get embedding for a sentence.

        Args:
            sentence: Sentence text

        Returns:
            Embedding vector
        """
        embedding = self.embedding_model.encode(
            [sentence],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding[0]

    def batch_search(
        self,
        queries: List[str],
        k: int = 10
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Batch search for multiple queries.

        Args:
            queries: List of query strings
            k: Number of results per query

        Returns:
            List of (distances, indices) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Get query embeddings
        query_embeddings = self.embedding_model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Search
        distances, indices = self.index.search(query_embeddings, k)

        # Return as list of tuples
        results = []
        for i in range(len(queries)):
            results.append((distances[i], indices[i]))

        return results


class FAISSRelevancePreFilter:
    """Fast relevance pre-filtering using FAISS."""

    def __init__(self, faiss_index: FAISSIndexBuilder):
        """
        Initialize pre-filter.

        Args:
            faiss_index: FAISS index builder
        """
        self.faiss_index = faiss_index
        self.embedding_model = faiss_index.embedding_model

    def pre_filter(
        self,
        evidence_hypotheses: List,  # List[EvidenceHypothesis]
        k: int = 50,
        similarity_threshold: float = 0.5
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Pre-filter sentences relevant to each hypothesis.

        Args:
            evidence_hypotheses: Evidence hypotheses
            k: Top-k candidates per hypothesis
            similarity_threshold: Minimum similarity

        Returns:
            {hypothesis_id: [(sentence, similarity), ...]}
        """
        logger.info(
            "Pre-filtering with FAISS",
            hypotheses=len(evidence_hypotheses),
            k=k
        )

        results = {}

        for hypothesis in evidence_hypotheses:
            # Search FAISS index
            distances, indices = self.faiss_index.search(
                hypothesis.hypothesis,
                k
            )

            # Filter by threshold and map to sentences
            candidates = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:  # Empty slot
                    continue

                similarity = float(dist)  # Cosine similarity

                if similarity >= similarity_threshold:
                    sentence = self.faiss_index.get_sentence(idx)
                    if sentence:
                        candidates.append((sentence, similarity))

            results[hypothesis.hypothesis_id] = candidates

        logger.info(
            "Pre-filtering completed",
            avg_candidates=sum(len(v) for v in results.values()) / max(1, len(results))
        )

        return results


def create_faiss_index(
    sentences: List[str],
    embedding_model: str = "all-MiniLM-L6-v2"
) -> FAISSIndexBuilder:
    """
    Create and build a FAISS index.

    Args:
        sentences: Sentences to index
        embedding_model: Embedding model name

    Returns:
        FAISSIndexBuilder with built index
    """
    builder = FAISSIndexBuilder(embedding_model=embedding_model)
    builder.build_index(sentences)
    return builder
