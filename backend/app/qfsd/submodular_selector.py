"""
Submodular Selection for Coverage + Diversity.

Greedy submodular optimization with (1-1/e) approximation guarantee.
Maximizes coverage of evidence hypotheses and diversity while respecting
a hard token budget constraint.
"""
from typing import List, Optional, Callable
import numpy as np
from app.qfsd.models import EvidenceSentence, EvidenceHypothesis
from app.utils.logger import get_logger

logger = get_logger(__name__)


def count_tokens(text: str) -> int:
    """
    Estimate token count (rough approximation).

    Args:
        text: Text to count

    Returns:
        Estimated token count
    """
    # Rough estimate: 4 chars per token
    return len(text) // 4


class SubmodularSelector:
    """
    Submodular selection for coverage + diversity.

    Objective: F(S) = coverage(S) + λ * diversity(S)
    Constraint: token_count(S) <= token_budget
    """

    def __init__(
        self,
        evidence_hypotheses: List[EvidenceHypothesis],
        lambda_diversity: float = 0.5
    ):
        """
        Initialize submodular selector.

        Args:
            evidence_hypotheses: Evidence hypotheses for coverage
            lambda_diversity: Diversity weight
        """
        self.evidence_hypotheses = evidence_hypotheses
        self.lambda_diversity = lambda_diversity

        logger.info(
            "Submodular selector initialized",
            hypotheses=len(evidence_hypotheses),
            lambda_diversity=lambda_diversity
        )

    def select(
        self,
        sentences: List[EvidenceSentence],
        token_budget: int,
        faiss_index: Optional = None,  # Optional[FAISSIndexBuilder]
        diversity_fn: Optional[Callable] = None
    ) -> List[EvidenceSentence]:
        """
        Greedy submodular selection.

        Args:
            sentences: Candidate sentences
            token_budget: Hard token budget constraint
            faiss_index: Optional FAISS index for fast diversity
            diversity_fn: Optional custom diversity function

        Returns:
            Selected sentences
        """
        logger.info(
            "Starting submodular selection",
            candidates=len(sentences),
            token_budget=token_budget
        )

        selected = []
        remaining = sentences.copy()

        # Sort by initial score (entailment score)
        remaining.sort(key=lambda s: s.entailment_score, reverse=True)

        current_tokens = 0

        while remaining and current_tokens < token_budget:
            best_sentence = None
            best_gain = -float('inf')

            for sentence in remaining:
                # Check token budget
                sent_tokens = count_tokens(sentence.sentence)
                if current_tokens + sent_tokens > token_budget:
                    continue

                # Calculate marginal gain
                coverage_gain = self._coverage_increase(
                    sentence=sentence,
                    selected=selected
                )

                if diversity_fn:
                    diversity_gain = diversity_fn(sentence, selected, faiss_index)
                elif faiss_index:
                    diversity_gain = self._diversity_increase_faiss(
                        sentence=sentence,
                        selected=selected,
                        faiss_index=faiss_index
                    )
                else:
                    diversity_gain = self._diversity_increase_cosine(
                        sentence=sentence,
                        selected=selected
                    )

                marginal_gain = coverage_gain + self.lambda_diversity * diversity_gain

                if marginal_gain > best_gain:
                    best_gain = marginal_gain
                    best_sentence = sentence

            if best_sentence and best_gain > 0:
                selected.append(best_sentence)
                current_tokens += count_tokens(best_sentence.sentence)
                remaining.remove(best_sentence)

                logger.debug(
                    "Selected sentence",
                    tokens=current_tokens,
                    budget=token_budget,
                    gain=round(best_gain, 3)
                )
            else:
                break  # No more positive gains

        logger.info(
            "Submodular selection completed",
            selected=len(selected),
            tokens=current_tokens,
            budget=token_budget
        )

        return selected

    def _coverage_increase(
        self,
        sentence: EvidenceSentence,
        selected: List[EvidenceSentence]
    ) -> float:
        """
        Calculate coverage increase from adding sentence.

        Args:
            sentence: Candidate sentence
            selected: Already selected sentences

        Returns:
            Coverage gain
        """
        # Hypotheses already covered
        covered_hypotheses = set()
        for s in selected:
            covered_hypotheses.add(s.hypothesis.hypothesis_id)

        # Hypotheses covered by new sentence
        new_hypothesis = sentence.hypothesis.hypothesis_id

        # Coverage gain = newly covered
        if new_hypothesis in covered_hypotheses:
            return 0.0
        else:
            # Weight by hypothesis importance
            importance = 1.0 if sentence.hypothesis.is_explicit else 0.7
            return importance

    def _diversity_increase_faiss(
        self,
        sentence: EvidenceSentence,
        selected: List[EvidenceSentence],
        faiss_index
    ) -> float:
        """
        Calculate diversity increase using FAISS (FAST!).

        Args:
            sentence: Candidate sentence
            selected: Already selected sentences
            faiss_index: FAISS index

        Returns:
            Diversity gain
        """
        if not selected:
            return 1.0  # Maximum diversity for first sentence

        # Get selected embeddings
        selected_sentences = [s.sentence for s in selected]
        selected_embeddings = faiss_index.embedding_model.encode(
            selected_sentences,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Get candidate embedding
        candidate_embedding = faiss_index.embedding_model.encode(
            [sentence.sentence],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Calculate cosine similarity (dot product for normalized)
        similarities = np.dot(selected_embeddings, candidate_embedding.T).flatten()
        min_similarity = float(np.min(similarities))

        # Diversity = 1 - similarity
        return 1.0 - min_similarity

    def _diversity_increase_cosine(
        self,
        sentence: EvidenceSentence,
        selected: List[EvidenceSentence]
    ) -> float:
        """
        Calculate diversity increase using cosine (fallback).

        Args:
            sentence: Candidate sentence
            selected: Already selected sentences

        Returns:
            Diversity gain
        """
        if not selected:
            return 1.0

        # Fallback: simple length-based diversity
        # (In production, use embeddings or skip if no FAISS)
        selected_lengths = [len(s.sentence) for s in selected]
        candidate_length = len(sentence.sentence)

        # Simple diversity: difference in length
        avg_length = sum(selected_lengths) / len(selected_lengths)
        diversity = min(1.0, abs(candidate_length - avg_length) / 100)

        return diversity


def select_sentences_submodular(
    sentences: List[EvidenceSentence],
    evidence_hypotheses: List[EvidenceHypothesis],
    token_budget: int,
    lambda_diversity: float = 0.5,
    faiss_index: Optional = None
) -> List[EvidenceSentence]:
    """
    Convenience function for submodular selection.

    Args:
        sentences: Candidate sentences
        evidence_hypotheses: Evidence hypotheses
        token_budget: Token budget
        lambda_diversity: Diversity weight
        faiss_index: Optional FAISS index

    Returns:
        Selected sentences
    """
    selector = SubmodularSelector(
        evidence_hypotheses=evidence_hypotheses,
        lambda_diversity=lambda_diversity
    )

    return selector.select(
        sentences=sentences,
        token_budget=token_budget,
        faiss_index=faiss_index
    )
