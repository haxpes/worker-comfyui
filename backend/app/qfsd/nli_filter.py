"""
Evidence-Aware Filtering using Natural Language Inference (NLI).

Filters sentences not just by topical/semantic similarity, but by whether
they actually support or answer the evidence hypotheses.
"""
from typing import List, Tuple, Dict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from app.qfsd.models import EvidenceHypothesis, EvidenceSentence, Contradiction
from app.utils.logger import get_logger

logger = get_logger(__name__)


class NLIFilter:
    """
    NLI-based evidence-aware filter.

    Uses NLI models to check if sentences entail (support) or contradict
    evidence hypotheses.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        domain: str = "general",
        device: str | None = None
    ):
        """
        Initialize NLI filter.

        Args:
            model_name: NLI model name
            domain: Domain for threshold calibration
            device: Device ("cuda" or "cpu")
        """
        self.model_name = model_name
        self.domain = domain

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(
            "Loading NLI model",
            model=model_name,
            device=self.device
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            logger.info("NLI model loaded successfully")

        except Exception as e:
            logger.error(
                "Failed to load NLI model",
                model=model_name,
                error=str(e)
            )
            raise

        # Domain-calibrated threshold
        self.threshold = self._get_domain_threshold(domain)

        logger.info(
            "NLI filter initialized",
            domain=domain,
            threshold=self.threshold
        )

    def _get_domain_threshold(self, domain: str) -> float:
        """
        Get calibrated threshold per domain.

        Args:
            domain: Domain name

        Returns:
            Threshold value
        """
        thresholds = {
            "general": 0.5,
            "regulatory": 0.6,  # Higher for regulatory (more precision)
            "medical": 0.55,
            "technical": 0.5
        }
        return thresholds.get(domain, 0.5)

    def _nli_check(
        self,
        sentence: str,
        hypothesis: str
    ) -> Tuple[float, float, float]:
        """
        Check NLI: entailment, contradiction, neutral.

        Args:
            sentence: Premise sentence
            hypothesis: Hypothesis (declarative statement)

        Returns:
            (entailment_prob, contradiction_prob, neutral_prob)
        """
        # Format for NLI: premise and hypothesis
        inputs = self.tokenizer(
            sentence,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

            # NLI labels: 0=contradiction, 1=neutral, 2=entailment
            contradiction_prob = probs[0][0].item()
            neutral_prob = probs[0][1].item()
            entailment_prob = probs[0][2].item()

        return entailment_prob, contradiction_prob, neutral_prob

    def _nli_check_batch(
        self,
        sentences: List[str],
        hypothesis: str,
        batch_size: int = 16
    ) -> List[Tuple[float, float, float]]:
        """
        Batch NLI check for efficiency.

        Args:
            sentences: List of premise sentences
            hypothesis: Single hypothesis
            batch_size: Batch size for inference

        Returns:
            List of (entailment, contradiction, neutral) tuples
        """
        results = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                [hypothesis] * len(batch),
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)

                # Extract probabilities
                for j in range(len(batch)):
                    contradiction_prob = probs[j][0].item()
                    neutral_prob = probs[j][1].item()
                    entailment_prob = probs[j][2].item()

                    results.append((
                        entailment_prob,
                        contradiction_prob,
                        neutral_prob
                    ))

        return results

    async def filter_evidence(
        self,
        sentences: List[str],
        evidence_hypotheses: List[EvidenceHypothesis],
        use_batch: bool = True
    ) -> Tuple[List[EvidenceSentence], List[Contradiction]]:
        """
        Filter sentences that actually support/answer evidence hypotheses.

        Args:
            sentences: Candidate sentences
            evidence_hypotheses: Evidence hypotheses to check against
            use_batch: Whether to use batch processing

        Returns:
            (evidence_sentences, contradictions)
        """
        logger.info(
            "Filtering evidence",
            sentences=len(sentences),
            hypotheses=len(evidence_hypotheses)
        )

        evidence_sentences = []
        contradictions = []
        seen_sentences = set()

        for hypothesis in evidence_hypotheses:
            if use_batch:
                # Batch NLI check
                results = self._nli_check_batch(
                    sentences,
                    hypothesis.hypothesis,
                    batch_size=16
                )

                for sentence, (entailment, contradiction, neutral) in zip(sentences, results):
                    if sentence in seen_sentences:
                        continue

                    if entailment > self.threshold:
                        # Sentence supports hypothesis
                        evidence_sentences.append(EvidenceSentence(
                            sentence=sentence,
                            hypothesis=hypothesis,
                            entailment_score=entailment,
                            evidence_type="support"
                        ))
                        seen_sentences.add(sentence)

                    elif contradiction > 0.5:
                        # Sentence contradicts hypothesis (flag for critique)
                        contradictions.append(Contradiction(
                            sentence=sentence,
                            hypothesis=hypothesis,
                            contradiction_score=contradiction
                        ))

            else:
                # Sequential NLI check
                for sentence in sentences:
                    if sentence in seen_sentences:
                        continue

                    entailment, contradiction, neutral = self._nli_check(
                        sentence,
                        hypothesis.hypothesis
                    )

                    if entailment > self.threshold:
                        evidence_sentences.append(EvidenceSentence(
                            sentence=sentence,
                            hypothesis=hypothesis,
                            entailment_score=entailment,
                            evidence_type="support"
                        ))
                        seen_sentences.add(sentence)

                    elif contradiction > 0.5:
                        contradictions.append(Contradiction(
                            sentence=sentence,
                            hypothesis=hypothesis,
                            contradiction_score=contradiction
                        ))

        logger.info(
            "Evidence filtering completed",
            evidence_sentences=len(evidence_sentences),
            contradictions=len(contradictions)
        )

        return evidence_sentences, contradictions

    async def filter_evidence_pre_filtered(
        self,
        pre_filtered_candidates: Dict[str, List[Tuple[str, float]]],
        evidence_hypotheses: List[EvidenceHypothesis],
        use_batch: bool = True
    ) -> Tuple[List[EvidenceSentence], List[Contradiction]]:
        """
        Filter evidence from pre-filtered FAISS candidates.

        Args:
            pre_filtered_candidates: {hypothesis_id: [(sentence, similarity), ...]}
            evidence_hypotheses: Evidence hypotheses
            use_batch: Whether to use batch processing

        Returns:
            (evidence_sentences, contradictions)
        """
        logger.info(
            "Filtering pre-filtered evidence",
            hypotheses=len(evidence_hypotheses)
        )

        evidence_sentences = []
        contradictions = []

        for hypothesis in evidence_hypotheses:
            candidates = pre_filtered_candidates.get(hypothesis.hypothesis_id, [])

            if not candidates:
                continue

            # Extract sentences
            sentences = [sent for sent, _ in candidates]

            if use_batch:
                # Batch NLI check
                results = self._nli_check_batch(
                    sentences,
                    hypothesis.hypothesis,
                    batch_size=16
                )

                for sentence, (entailment, contradiction, neutral) in zip(sentences, results):
                    if entailment > self.threshold:
                        evidence_sentences.append(EvidenceSentence(
                            sentence=sentence,
                            hypothesis=hypothesis,
                            entailment_score=entailment,
                            evidence_type="support"
                        ))

                    elif contradiction > 0.5:
                        contradictions.append(Contradiction(
                            sentence=sentence,
                            hypothesis=hypothesis,
                            contradiction_score=contradiction
                        ))

            else:
                # Sequential
                for sentence, _ in candidates:
                    entailment, contradiction, neutral = self._nli_check(
                        sentence,
                        hypothesis.hypothesis
                    )

                    if entailment > self.threshold:
                        evidence_sentences.append(EvidenceSentence(
                            sentence=sentence,
                            hypothesis=hypothesis,
                            entailment_score=entailment,
                            evidence_type="support"
                        ))

                    elif contradiction > 0.5:
                        contradictions.append(Contradiction(
                            sentence=sentence,
                            hypothesis=hypothesis,
                            contradiction_score=contradiction
                        ))

        logger.info(
            "Pre-filtered evidence filtering completed",
            evidence_sentences=len(evidence_sentences),
            contradictions=len(contradictions)
        )

        return evidence_sentences, contradictions


def create_nli_filter(
    model_name: str = "cross-encoder/nli-deberta-v3-base",
    domain: str = "general"
) -> NLIFilter:
    """
    Create an NLI filter instance.

    Args:
        model_name: NLI model name
        domain: Domain for threshold calibration

    Returns:
        NLIFilter
    """
    return NLIFilter(model_name=model_name, domain=domain)
