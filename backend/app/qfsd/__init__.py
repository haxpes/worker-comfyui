"""
QFSD (Query-Focused Summary Document) Pipeline.

A comprehensive information retrieval pipeline that follows:
Wide Search → Relevance Filtering → Deduplication

Core principle: Cast a wide net, then intelligently filter to keep
only what's needed and relevant.
"""

from app.qfsd.models import (
    InformationNeeds,
    EvidenceHypothesis,
    QDMRPlan,
    RelevanceCriteria,
    EvidenceSentence,
    CoverageReport,
    QFSD
)

__all__ = [
    "InformationNeeds",
    "EvidenceHypothesis",
    "QDMRPlan",
    "RelevanceCriteria",
    "EvidenceSentence",
    "CoverageReport",
    "QFSD"
]
