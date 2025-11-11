"""
QFSD Agent: Query-Focused Summary Document generation with LangGraph.

Implements a comprehensive retrieval agent using the QFSD pipeline for
high-quality, token-efficient information retrieval with coverage tracking.
"""
import time
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from app.config import settings
from app.core.state import AgentState
from app.agents.base import BaseAgent
from app.retrieval.qfsd import create_qfsd_retriever
from app.utils.prompts import extract_citations_from_answer
from app.utils.logger import get_logger

logger = get_logger(__name__)


SYNTHESIS_PROMPT = """You are an expert at synthesizing information into comprehensive answers.

Original Question: {query}

Context (Query-Focused Summary Document):
{context}

Coverage Information:
{coverage_info}

Task: Provide a comprehensive answer to the original question using the provided context.

Requirements:
1. Use ALL relevant information from the context
2. Cite sources using [N] notation
3. Maintain logical flow and coherence
4. Address all aspects of the question
5. Note any contradictions found: {contradictions}

Comprehensive Answer:"""


class QFSDAgent(BaseAgent):
    """
    QFSD Agent for comprehensive information retrieval.

    Uses the 15-stage QFSD pipeline:
    1-4. Query Analysis & Evidence Hypotheses
    5. Wide Search
    6. MMR Re-ranking
    7. Sentence Splitting (spaCy)
    8. Build FAISS Index
    9. Fast Pre-filtering (FAISS)
    10. Evidence-Aware Filtering (NLI)
    11. Coverage Tracking
    12. Fast Deduplication (FAISS)
    13. Submodular Selection
    14. Sentence Combination
    15. Build QFSD

    Workflow:
    - build_qfsd: Execute full QFSD pipeline
    - synthesize: Generate answer from QFSD
    """

    def __init__(self, corpus_id: str | None = None):
        """
        Initialize QFSD agent.

        Args:
            corpus_id: Optional corpus ID
        """
        super().__init__(name="qfsd")

        # Initialize QFSD retriever
        self.qfsd_retriever = create_qfsd_retriever(
            corpus_id=corpus_id,
            wide_search_k=200,
            mmr_k=80,
            token_budget=4000,
            domain="general",
            use_raptor=bool(corpus_id)
        )

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.complex_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )

        logger.info("QFSD Agent initialized", corpus_id=corpus_id)

    def build_graph(self) -> CompiledGraph:
        """
        Build the LangGraph workflow.

        Nodes:
        - build_qfsd: Execute full QFSD pipeline
        - synthesize: Generate answer from QFSD

        Returns:
            Compiled LangGraph workflow
        """
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("build_qfsd", self._build_qfsd_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Define edges
        workflow.set_entry_point("build_qfsd")
        workflow.add_edge("build_qfsd", "synthesize")
        workflow.add_edge("synthesize", END)

        logger.info("QFSD Agent graph built")

        return workflow.compile()

    async def _build_qfsd_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Build QFSD document using full pipeline.

        Args:
            state: Current agent state

        Returns:
            Updated state with QFSD
        """
        query = state["query"]
        start_time = time.time()

        logger.info("build_qfsd node started", query_preview=query[:100])

        try:
            # Execute full QFSD pipeline
            qfsd = await self.qfsd_retriever.retrieve(query)

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                "build_qfsd node completed",
                sentences=len(qfsd.sentences),
                coverage=round(qfsd.coverage_report.overall_coverage, 2) if qfsd.coverage_report else 0,
                duration_ms=round(duration_ms, 2)
            )

            return {
                "metadata": {
                    "qfsd": qfsd,
                    "qfsd_context": qfsd.to_prompt_context(),
                    "coverage_report": qfsd.coverage_report,
                    "contradictions": qfsd.contradictions
                }
            }

        except Exception as e:
            logger.error("build_qfsd node failed", error=str(e))
            raise

    async def _synthesize_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Synthesize final answer from QFSD.

        Args:
            state: Current agent state

        Returns:
            Updated state with answer
        """
        query = state["query"]
        metadata = state.get("metadata", {})
        qfsd = metadata.get("qfsd")
        qfsd_context = metadata.get("qfsd_context", "")
        coverage_report = metadata.get("coverage_report")
        contradictions = metadata.get("contradictions", [])

        start_time = time.time()

        logger.info("synthesize node started")

        try:
            # Prepare coverage information
            coverage_info = ""
            if coverage_report:
                coverage_info = f"Overall Coverage: {coverage_report.overall_coverage:.2%}"
                if coverage_report.gaps:
                    coverage_info += f"\nGaps: {', '.join(coverage_report.gaps)}"

            # Prepare contradiction information
            contradiction_info = ""
            if contradictions:
                contradiction_info = f"{len(contradictions)} contradictions detected"

            # Synthesize answer
            prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            answer = await chain.ainvoke({
                "query": query,
                "context": qfsd_context,
                "coverage_info": coverage_info,
                "contradictions": contradiction_info
            })

            # Extract citations
            citations = []
            if qfsd:
                # Create pseudo-documents from QFSD sentences for citation
                from langchain_core.documents import Document
                pseudo_docs = [
                    Document(
                        page_content=sent,
                        metadata={"id": f"qfsd_sent_{i}", "type": "qfsd_sentence"}
                    )
                    for i, sent in enumerate(qfsd.sentences)
                ]
                citations = extract_citations_from_answer(answer, pseudo_docs)

            # Confidence based on coverage
            if coverage_report:
                confidence = coverage_report.overall_coverage * 0.9  # Scale down slightly
            else:
                confidence = 0.7

            # Reduce confidence if contradictions found
            if contradictions:
                confidence *= 0.9

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                "synthesize node completed",
                answer_length=len(answer),
                citations=len(citations),
                confidence=round(confidence, 2),
                duration_ms=round(duration_ms, 2)
            )

            return {
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "reranked_docs": []  # QFSD doesn't return docs directly
            }

        except Exception as e:
            logger.error("synthesize node failed", error=str(e))
            raise


def create_qfsd_agent(corpus_id: str | None = None) -> QFSDAgent:
    """
    Create a QFSD agent instance.

    Args:
        corpus_id: Optional corpus ID

    Returns:
        QFSDAgent
    """
    return QFSDAgent(corpus_id=corpus_id)
