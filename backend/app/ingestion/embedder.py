"""
OpenAI embedding generation with batching and retry logic.
"""
import asyncio
from typing import List
import numpy as np
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """
    OpenAI embedding generator with batching and error handling.
    """

    def __init__(self):
        """Initialize embedder with OpenAI client."""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self.dimension = settings.embedding_dimension
        self.batch_size = 100  # OpenAI allows up to 2048 texts per request

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            Exception if embedding generation fails after retries
        """
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=self.model
            )
            embedding = response.data[0].embedding

            logger.debug(
                "Generated embedding",
                text_length=len(text),
                embedding_dim=len(embedding)
            )

            return embedding

        except Exception as e:
            logger.error(
                "Failed to generate embedding",
                error=str(e),
                text_preview=text[:100]
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            Exception if embedding generation fails after retries
        """
        if not texts:
            return []

        try:
            response = await self.client.embeddings.create(
                input=texts,
                model=self.model
            )

            # Sort by index to ensure correct order
            embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

            logger.info(
                "Generated batch embeddings",
                batch_size=len(texts),
                embedding_dim=len(embeddings[0]) if embeddings else 0
            )

            return embeddings

        except Exception as e:
            logger.error(
                "Failed to generate batch embeddings",
                error=str(e),
                batch_size=len(texts)
            )
            raise

    async def embed_documents(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for a large list of texts with batching.

        Args:
            texts: List of texts to embed
            show_progress: Whether to log progress

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        logger.info(
            "Starting embedding generation",
            total_texts=len(texts),
            batch_size=self.batch_size,
            total_batches=total_batches
        )

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            if show_progress:
                logger.info(
                    f"Processing batch {batch_num}/{total_batches}",
                    batch_size=len(batch),
                    progress_pct=round(batch_num / total_batches * 100, 1)
                )

            batch_embeddings = await self.embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

            # Small delay to avoid rate limits
            if i + self.batch_size < len(texts):
                await asyncio.sleep(0.1)

        logger.info(
            "Embedding generation completed",
            total_embeddings=len(all_embeddings)
        )

        return all_embeddings

    def cosine_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score (0 to 1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


# Global embedder instance
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """
    Get global embedder instance (singleton).

    Returns:
        Embedder instance
    """
    global _embedder

    if _embedder is None:
        _embedder = Embedder()
        logger.info(
            "Embedder initialized",
            model=_embedder.model,
            dimension=_embedder.dimension
        )

    return _embedder
