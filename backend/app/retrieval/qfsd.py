"""
QFSD Retriever: Query-Focused Summary Document generation.

Implements the comprehensive 15-stage pipeline:
Wide Search → Relevance Filtering → Deduplication
"""
import time
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from app.config import settings
from app.retrieval.hybrid import create_hybrid_retriever
from app.retrieval.raptor import create_raptor_retriever
from app.retrieval.reranker import rerank_documents
from app.qfsd.models import (
    InformationNeeds,
    QFSD,
    EvidenceSentence
)
from app.qfsd.qdmr_decomposer import create_qdmr_decomposer
from app.qfsd.nli_filter import create_nli_filter
from app.qfsd.faiss_indexer import create_faiss_index, FAISSRelevancePreFilter
from app.qfsd.submodular_selector import select_sentences_submodular
from app.qfsd.coverage_tracker import create_coverage_tracker
from app.qfsd.sentence_combiner import split_into_sentences, create_sentence_combiner
from app.utils.logger import get_logger

logger = get_logger(__name__)


class QFSDRetriever:
    """
    QFSD Retriever implementing the full 15-stage pipeline.

    Pipeline:
    1-4. Query Analysis & Evidence Hypothesis Generation
    5. Wide Search (intelligently guided)
    6. MMR Re-ranking (diversity before sentence splitting)
    7. Sentence Splitting (spaCy)
    8. Build FAISS Index
    9. Fast Relevance Pre-filtering (FAISS)
    10. Evidence-Aware Filtering (NLI)
    11. Coverage Tracking
    12. Fast Deduplication (FAISS)
    13. Submodular Selection
    14. Sentence Combination
    15. Build QFSD
    """

    def __init__(
        self,
        corpus_id: Optional[str] = None,
        wide_search_k: int = 200,
        mmr_k: int = 80,
        token_budget: int = 4000,
        domain: str = "general",
        use_raptor: bool = True,
        max_iterations: int = 3
    ):
        """
        Initialize QFSD retriever.

        Args:
            corpus_id: Optional corpus ID
            wide_search_k: Initial wide search k
            mmr_k: MMR re-ranking k
            token_budget: Hard token budget for QFSD
            domain: Domain for NLI threshold calibration
            use_raptor: Whether to use RAPTOR analysis
            max_iterations: Max iterations for iterative retrieval
        """
        self.corpus_id = corpus_id
        self.wide_search_k = wide_search_k
        self.mmr_k = mmr_k
        self.token_budget = token_budget
        self.domain = domain
        self.use_raptor = use_raptor
        self.max_iterations = max_iterations

        # Initialize components
        self.hybrid_retriever = create_hybrid_retriever(corpus_id=corpus_id)
        self.qdmr_decomposer = create_qdmr_decomposer()
        self.nli_filter = create_nli_filter(domain=domain)

        if use_raptor and corpus_id:
            self.raptor_retriever = create_raptor_retriever(
                corpus_id=corpus_id,
                k=10,
                expand_to_chunks=False
            )
        else:
            self.raptor_retriever = None

        logger.info(
            "QFSD Retriever initialized",
            corpus_id=corpus_id,
            wide_search_k=wide_search_k,
            mmr_k=mmr_k,
            token_budget=token_budget,
            domain=domain
        )

    async def retrieve(
        self,
        query: str,
        **kwargs
    ) -> QFSD:
        """
        Execute full QFSD pipeline.

        Args:
            query: Search query
            **kwargs: Additional parameters

        Returns:
            QFSD document
        """
        logger.info("Starting QFSD retrieval", query_preview=query[:100])
        start_time = time.time()

        try:
            # Stage 1-2: Query Analysis & QDMR Decomposition
            logger.info("Stage 1-2: QDMR decomposition")
            qdmr_plan = await self.qdmr_decomposer.decompose(query)
            evidence_hypotheses = qdmr_plan.evidence_hypotheses

            # Stage 3: RAPTOR Summary Analysis (optional)
            raptor_docs = []
            if self.raptor_retriever:
                logger.info("Stage 3: RAPTOR summary analysis")
                raptor_docs = await self.raptor_retriever.retrieve(query)

            # Stage 4: Define Relevance Criteria (implicit in hypotheses)
            logger.info("Stage 4: Relevance criteria defined from hypotheses")

            # Stage 5: Wide Search (guided by hypotheses)
            logger.info("Stage 5: Wide search", k=self.wide_search_k)
            documents = await self._wide_search(query, qdmr_plan, raptor_docs)

            # Stage 6: MMR Re-ranking (diversity before sentence splitting)
            logger.info("Stage 6: MMR re-ranking", target_k=self.mmr_k)
            documents = await rerank_documents(
                query=query,
                documents=documents,
                top_k=self.mmr_k
            )

            # Stage 7: Sentence Splitting (spaCy)
            logger.info("Stage 7: Sentence splitting with spaCy")
            sentences = []
            for doc in documents:
                doc_sentences = split_into_sentences(doc.page_content)
                sentences.extend(doc_sentences)

            logger.info("Sentences extracted", count=len(sentences))

            # Stage 8: Build FAISS Index
            logger.info("Stage 8: Building FAISS index")
            faiss_index = create_faiss_index(sentences)

            # Stage 9: Fast Relevance Pre-filtering (FAISS)
            logger.info("Stage 9: FAISS pre-filtering")
            pre_filter = FAISSRelevancePreFilter(faiss_index)
            pre_filtered_candidates = pre_filter.pre_filter(
                evidence_hypotheses=evidence_hypotheses,
                k=50,
                similarity_threshold=0.5
            )

            # Stage 10: Evidence-Aware Filtering (NLI on pre-filtered)
            logger.info("Stage 10: Evidence-aware filtering (NLI)")
            evidence_sentences, contradictions = await self.nli_filter.filter_evidence_pre_filtered(
                pre_filtered_candidates=pre_filtered_candidates,
                evidence_hypotheses=evidence_hypotheses,
                use_batch=True
            )

            # Stage 11: Coverage Tracking
            logger.info("Stage 11: Coverage tracking")
            coverage_tracker = create_coverage_tracker(evidence_hypotheses)
            coverage_report = coverage_tracker.update_coverage(evidence_sentences)

            # Iterative retrieval if gaps found
            iteration = 0
            while not coverage_report.is_complete() and iteration < self.max_iterations:
                iteration += 1
                logger.info(f"Iteration {iteration}: Retrieving for gaps")

                gap_evidence = await self._retrieve_for_gaps(
                    query=query,
                    gaps=coverage_report.gaps,
                    evidence_hypotheses=evidence_hypotheses,
                    faiss_index=faiss_index
                )

                evidence_sentences.extend(gap_evidence)
                coverage_report = coverage_tracker.update_coverage(evidence_sentences)

            # Stage 12: Fast Deduplication (FAISS)
            logger.info("Stage 12: FAISS deduplication")
            unique_sentences = faiss_index.deduplicate_sentences(
                sentences=[s.sentence for s in evidence_sentences],
                similarity_threshold=0.85
            )

            # Map back to EvidenceSentence objects
            unique_evidence = [
                s for s in evidence_sentences
                if s.sentence in unique_sentences
            ]

            # Stage 13: Submodular Selection
            logger.info(
                "Stage 13: Submodular selection",
                budget=self.token_budget
            )
            selected_sentences = select_sentences_submodular(
                sentences=unique_evidence,
                evidence_hypotheses=evidence_hypotheses,
                token_budget=self.token_budget,
                lambda_diversity=0.5,
                faiss_index=faiss_index
            )

            # Stage 14: Sentence Combination
            logger.info("Stage 14: Sentence combination")
            sentence_combiner = create_sentence_combiner(
                faiss_index=faiss_index
            )
            combined_sentences = sentence_combiner.combine_sentences(
                sentences=[s.sentence for s in selected_sentences],
                similarity_threshold=0.7
            )

            # Stage 15: Build QFSD
            logger.info("Stage 15: Building QFSD")
            qfsd = QFSD(
                query=query,
                sentences=combined_sentences,
                source_mapping={},  # TODO: track source documents
                information_needs_coverage={},  # TODO: track coverage mapping
                relevance_scores={},
                coverage_report=coverage_report,
                contradictions=contradictions,
                metadata={
                    "evidence_hypotheses": len(evidence_hypotheses),
                    "unique_sentences": len(unique_evidence),
                    "selected_sentences": len(selected_sentences),
                    "combined_sentences": len(combined_sentences),
                    "iterations": iteration
                }
            )

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                "QFSD retrieval completed",
                sentences=len(qfsd.sentences),
                duration_ms=round(duration_ms, 2),
                coverage=round(coverage_report.overall_coverage, 2)
            )

            return qfsd

        except Exception as e:
            logger.error("QFSD retrieval failed", error=str(e))
            raise

    async def _wide_search(
        self,
        query: str,
        qdmr_plan,
        raptor_docs: List[Document]
    ) -> List[Document]:
        """
        Wide search guided by QDMR plan and RAPTOR insights.

        Args:
            query: Original query
            qdmr_plan: QDMR plan with sub-questions
            raptor_docs: RAPTOR summary documents

        Returns:
            Retrieved documents
        """
        all_docs = []

        # Search with original query
        docs = await self.hybrid_retriever.retrieve(query)
        all_docs.extend(docs)

        # Search with each sub-question
        for subq in qdmr_plan.sub_questions:
            docs = await self.hybrid_retriever.retrieve(subq.question)
            all_docs.extend(docs)

        # Add RAPTOR docs
        all_docs.extend(raptor_docs)

        # Deduplicate by doc ID
        seen_ids = set()
        unique_docs = []
        for doc in all_docs:
            doc_id = doc.metadata.get("id")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)

        logger.info(
            "Wide search completed",
            total_docs=len(all_docs),
            unique_docs=len(unique_docs)
        )

        return unique_docs[:self.wide_search_k]

    async def _retrieve_for_gaps(
        self,
        query: str,
        gaps: List[str],
        evidence_hypotheses: List,
        faiss_index
    ) -> List[EvidenceSentence]:
        """
        Retrieve additional evidence for gaps.

        Args:
            query: Original query
            gaps: Information needs with gaps
            evidence_hypotheses: Evidence hypotheses
            faiss_index: FAISS index

        Returns:
            Additional evidence sentences
        """
        gap_hypotheses = [
            h for h in evidence_hypotheses
            if h.information_need in gaps
        ]

        if not gap_hypotheses:
            return []

        # Retrieve with gap-focused queries
        all_docs = []
        for hypothesis in gap_hypotheses:
            docs = await self.hybrid_retriever.retrieve(hypothesis.hypothesis)
            all_docs.extend(docs[:10])

        # MMR re-ranking
        reranked = await rerank_documents(
            query=query,
            documents=all_docs,
            top_k=20
        )

        # Sentence splitting
        gap_sentences = []
        for doc in reranked:
            doc_sentences = split_into_sentences(doc.page_content)
            gap_sentences.extend(doc_sentences)

        # Update FAISS index (rebuild with new sentences)
        # Note: In production, consider incremental updates
        all_sentences = list(faiss_index.sentence_map.values()) + gap_sentences
        faiss_index.build_index(all_sentences)

        # Pre-filter and NLI check
        pre_filter = FAISSRelevancePreFilter(faiss_index)
        gap_pre_filtered = pre_filter.pre_filter(
            evidence_hypotheses=gap_hypotheses,
            k=50,
            similarity_threshold=0.5
        )

        gap_evidence, _ = await self.nli_filter.filter_evidence_pre_filtered(
            pre_filtered_candidates=gap_pre_filtered,
            evidence_hypotheses=gap_hypotheses,
            use_batch=True
        )

        logger.info("Gap retrieval completed", evidence=len(gap_evidence))

        return gap_evidence


def create_qfsd_retriever(
    corpus_id: Optional[str] = None,
    wide_search_k: int = 200,
    mmr_k: int = 80,
    token_budget: int = 4000,
    domain: str = "general",
    use_raptor: bool = True
) -> QFSDRetriever:
    """
    Create a QFSD retriever instance.

    Args:
        corpus_id: Optional corpus ID
        wide_search_k: Initial wide search k
        mmr_k: MMR re-ranking k
        token_budget: Hard token budget
        domain: Domain for NLI calibration
        use_raptor: Whether to use RAPTOR

    Returns:
        QFSDRetriever
    """
    return QFSDRetriever(
        corpus_id=corpus_id,
        wide_search_k=wide_search_k,
        mmr_k=mmr_k,
        token_budget=token_budget,
        domain=domain,
        use_raptor=use_raptor
    )
