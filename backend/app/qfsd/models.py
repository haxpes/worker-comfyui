"""
Pydantic models for QFSD pipeline.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class RelevanceCriteria(BaseModel):
    """Criteria for determining relevance."""
    topical_keywords: List[str] = Field(
        default_factory=list,
        description="Key entities/concepts"
    )
    semantic_themes: List[str] = Field(
        default_factory=list,
        description="Semantic concepts"
    )
    factual_requirements: List[str] = Field(
        default_factory=list,
        description="Required facts"
    )
    contextual_needs: List[str] = Field(
        default_factory=list,
        description="Context requirements"
    )
    scoring_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "topical": 0.3,
            "semantic": 0.3,
            "factual": 0.3,
            "contextual": 0.1
        },
        description="Dimension weights"
    )
    threshold: float = Field(
        default=0.5,
        description="Minimum relevance score"
    )


class InformationNeeds(BaseModel):
    """Information needs extracted from query."""
    query: str
    intent: str = Field(
        description="Query intent: factual, conceptual, comparative, procedural"
    )
    explicit_needs: List[str] = Field(
        default_factory=list,
        description="Directly mentioned needs"
    )
    implicit_needs: List[str] = Field(
        default_factory=list,
        description="Required but not stated needs"
    )
    supporting_needs: List[str] = Field(
        default_factory=list,
        description="Helpful context"
    )
    relevance_criteria: RelevanceCriteria = Field(
        default_factory=RelevanceCriteria
    )

    @property
    def all_needs(self) -> List[str]:
        """Get all needs combined."""
        return self.explicit_needs + self.implicit_needs + self.supporting_needs


class SubQuestion(BaseModel):
    """QDMR sub-question."""
    subq_id: str
    question: str
    reasoning: str
    info_needs: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(
        default_factory=list,
        description="Other sub-question IDs this depends on"
    )


class EvidenceHypothesis(BaseModel):
    """Declarative evidence hypothesis for NLI checking."""
    hypothesis_id: str
    hypothesis: str = Field(
        description="Declarative statement for NLI check"
    )
    information_need: str = Field(
        description="Original information need"
    )
    sub_question: str = Field(
        description="Original QDMR sub-question"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="Other hypothesis IDs this depends on"
    )
    is_explicit: bool = Field(
        default=True,
        description="Whether this is an explicit need"
    )


class QDMRPlan(BaseModel):
    """QDMR decomposition plan."""
    query: str
    sub_questions: List[SubQuestion]
    evidence_hypotheses: List[EvidenceHypothesis]
    dependency_graph: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="DAG of dependencies"
    )


class RAPTORAnalysis(BaseModel):
    """RAPTOR summary analysis results."""
    summaries: List[Dict[str, Any]] = Field(default_factory=list)
    topics_covered: List[str] = Field(default_factory=list)
    detail_requirements: Dict[str, str] = Field(
        default_factory=dict,
        description="hypothesis_id -> detail level (summary|chunk)"
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="Topics not covered in summaries"
    )


class EvidenceSentence(BaseModel):
    """Sentence with evidence metadata."""
    sentence: str
    hypothesis: EvidenceHypothesis
    entailment_score: float
    evidence_type: str = Field(
        default="support",
        description="support, contradict, or neutral"
    )
    source_doc_id: Optional[str] = None
    embedding: Optional[List[float]] = None


class Contradiction(BaseModel):
    """Contradictory sentence flagged for critique."""
    sentence: str
    hypothesis: EvidenceHypothesis
    contradiction_score: float
    source_doc_id: Optional[str] = None


class CoverageData(BaseModel):
    """Coverage data for a single information need."""
    status: str = Field(
        description="missing, partial, or satisfied"
    )
    coverage_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )
    supporting_sentences: List[EvidenceSentence] = Field(default_factory=list)
    sentence_count: int = 0


class CoverageReport(BaseModel):
    """Coverage report for all information needs."""
    coverage: Dict[str, CoverageData] = Field(default_factory=dict)
    gaps: List[str] = Field(
        default_factory=list,
        description="Needs with insufficient coverage"
    )
    overall_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    def is_complete(self, threshold: float = 0.8) -> bool:
        """Check if coverage is complete."""
        return self.overall_coverage >= threshold and len(self.gaps) == 0


class QFSD(BaseModel):
    """Query-Focused Summary Document."""
    query: str
    sentences: List[str] = Field(
        description="Organized, deduplicated, relevant sentences"
    )
    source_mapping: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="sentence -> [doc_ids]"
    )
    information_needs_coverage: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="need -> [sentence_ids]"
    )
    relevance_scores: Dict[str, float] = Field(default_factory=dict)
    coverage_report: Optional[CoverageReport] = None
    contradictions: List[Contradiction] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_prompt_context(self) -> str:
        """Convert to string for LLM context."""
        return "\n\n".join(self.sentences)

    def get_token_count(self) -> int:
        """Estimate token count (rough approximation)."""
        # Rough estimate: 4 chars per token
        return len(self.to_prompt_context()) // 4
