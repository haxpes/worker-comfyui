"""
Coverage Tracking for Information Needs.

Tracks which evidence hypotheses are covered by retrieved sentences
and identifies gaps that require additional retrieval.
"""
from typing import List
from app.qfsd.models import (
    EvidenceHypothesis,
    EvidenceSentence,
    CoverageReport,
    CoverageData
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CoverageTracker:
    """
    Track coverage of evidence hypotheses.

    Provides:
    - Per-hypothesis coverage tracking
    - Gap detection
    - Coverage quality assessment
    """

    def __init__(self, evidence_hypotheses: List[EvidenceHypothesis]):
        """
        Initialize coverage tracker.

        Args:
            evidence_hypotheses: Evidence hypotheses to track
        """
        self.hypotheses = evidence_hypotheses
        self.coverage = {
            h.hypothesis_id: CoverageData(
                status="missing",
                coverage_score=0.0,
                supporting_sentences=[],
                sentence_count=0
            )
            for h in evidence_hypotheses
        }

        logger.info(
            "Coverage tracker initialized",
            hypotheses=len(evidence_hypotheses)
        )

    def update_coverage(
        self,
        evidence_sentences: List[EvidenceSentence]
    ) -> CoverageReport:
        """
        Update coverage based on evidence sentences.

        Args:
            evidence_sentences: Evidence sentences with hypothesis links

        Returns:
            Coverage report
        """
        logger.info(
            "Updating coverage",
            evidence_sentences=len(evidence_sentences)
        )

        # Reset counts
        for data in self.coverage.values():
            data.supporting_sentences = []
            data.sentence_count = 0

        # Track sentences per hypothesis
        for evidence in evidence_sentences:
            hypothesis_id = evidence.hypothesis.hypothesis_id

            if hypothesis_id in self.coverage:
                self.coverage[hypothesis_id].supporting_sentences.append(evidence)
                self.coverage[hypothesis_id].sentence_count += 1

        # Calculate coverage scores
        for hypothesis_id, data in self.coverage.items():
            sentence_count = data.sentence_count

            if sentence_count > 0:
                # Coverage score based on:
                # 1. Number of supporting sentences (up to 5 is good)
                # 2. Average entailment score
                sentence_factor = min(1.0, sentence_count / 5.0)

                avg_entailment = sum(
                    s.entailment_score
                    for s in data.supporting_sentences
                ) / sentence_count

                # Combined score
                data.coverage_score = min(
                    1.0,
                    0.4 * sentence_factor + 0.6 * avg_entailment
                )

                # Determine status
                if data.coverage_score >= 0.8:
                    data.status = "satisfied"
                elif data.coverage_score >= 0.5:
                    data.status = "partial"
                else:
                    data.status = "missing"
            else:
                data.coverage_score = 0.0
                data.status = "missing"

        # Generate report
        report = self._generate_report()

        logger.info(
            "Coverage updated",
            overall_coverage=round(report.overall_coverage, 2),
            gaps=len(report.gaps)
        )

        return report

    def _generate_report(self) -> CoverageReport:
        """
        Generate coverage report.

        Returns:
            CoverageReport
        """
        # Identify gaps
        gaps = []
        for hypothesis_id, data in self.coverage.items():
            if data.status in ["missing", "partial"]:
                hypothesis = self._get_hypothesis(hypothesis_id)
                if hypothesis:
                    gaps.append(hypothesis.information_need)

        # Calculate overall coverage
        overall_coverage = self._calculate_overall_coverage()

        return CoverageReport(
            coverage=self.coverage.copy(),
            gaps=gaps,
            overall_coverage=overall_coverage
        )

    def _calculate_overall_coverage(self) -> float:
        """
        Calculate overall coverage score.

        Returns:
            Overall coverage (0-1)
        """
        if not self.coverage:
            return 0.0

        # Weight explicit hypotheses more heavily
        explicit_hypotheses = [
            h for h in self.hypotheses if h.is_explicit
        ]
        implicit_hypotheses = [
            h for h in self.hypotheses if not h.is_explicit
        ]

        # Explicit coverage
        if explicit_hypotheses:
            explicit_score = sum(
                self.coverage[h.hypothesis_id].coverage_score
                for h in explicit_hypotheses
            ) / len(explicit_hypotheses)
        else:
            explicit_score = 1.0

        # Implicit coverage
        if implicit_hypotheses:
            implicit_score = sum(
                self.coverage[h.hypothesis_id].coverage_score
                for h in implicit_hypotheses
            ) / len(implicit_hypotheses)
        else:
            implicit_score = 1.0

        # Weighted average (70% explicit, 30% implicit)
        overall = 0.7 * explicit_score + 0.3 * implicit_score

        return overall

    def _get_hypothesis(self, hypothesis_id: str) -> EvidenceHypothesis | None:
        """
        Get hypothesis by ID.

        Args:
            hypothesis_id: Hypothesis ID

        Returns:
            Hypothesis or None
        """
        return next(
            (h for h in self.hypotheses if h.hypothesis_id == hypothesis_id),
            None
        )

    def get_gaps(self) -> List[str]:
        """
        Get information needs with gaps.

        Returns:
            List of information needs
        """
        gaps = []
        for hypothesis_id, data in self.coverage.items():
            if data.status in ["missing", "partial"]:
                hypothesis = self._get_hypothesis(hypothesis_id)
                if hypothesis:
                    gaps.append(hypothesis.information_need)
        return gaps

    def is_complete(self, threshold: float = 0.8) -> bool:
        """
        Check if coverage is complete.

        Args:
            threshold: Coverage threshold

        Returns:
            True if complete
        """
        overall = self._calculate_overall_coverage()
        gaps = self.get_gaps()

        return overall >= threshold and len(gaps) == 0

    def get_coverage_summary(self) -> dict:
        """
        Get coverage summary.

        Returns:
            Summary dict
        """
        satisfied = sum(
            1 for data in self.coverage.values()
            if data.status == "satisfied"
        )
        partial = sum(
            1 for data in self.coverage.values()
            if data.status == "partial"
        )
        missing = sum(
            1 for data in self.coverage.values()
            if data.status == "missing"
        )

        return {
            "total_hypotheses": len(self.hypotheses),
            "satisfied": satisfied,
            "partial": partial,
            "missing": missing,
            "overall_coverage": self._calculate_overall_coverage()
        }


def create_coverage_tracker(
    evidence_hypotheses: List[EvidenceHypothesis]
) -> CoverageTracker:
    """
    Create a coverage tracker instance.

    Args:
        evidence_hypotheses: Evidence hypotheses to track

    Returns:
        CoverageTracker
    """
    return CoverageTracker(evidence_hypotheses=evidence_hypotheses)
