"""
RAPTORAgent: Uses hierarchical RAPTOR summaries for topical retrieval.
Best for long documents with clear topical structure.
"""
import time
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from app.config import settings
from app.core.state import AgentState
from app.agents.base import BaseAgent
from app.retrieval.raptor import create_raptor_retriever
from app.retrieval.reranker import rerank_documents
from app.utils.prompts import get_synthesis_prompt, format_context_with_citations, extract_citations_from_answer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAPTORAgent(BaseAgent):
    """
    RAPTORAgent uses hierarchical summaries for topical retrieval.

    Workflow:
    1. Retrieve: Search RAPTOR summaries (all levels)
    2. Expand: Expand summaries to underlying chunks
    3. Rerank: Cross-encoder reranking
    4. Synthesize: LLM generates answer with hierarchy context
    """

    def __init__(self, corpus_id: str | None = None):
        """
        Initialize RAPTORAgent.

        Args:
            corpus_id: Corpus ID for RAPTOR retrieval (required)
        """
        super().__init__(name="raptor")

        if not corpus_id:
            raise ValueError("corpus_id is required for RAPTORAgent")

        self.corpus_id = corpus_id

        # Initialize retriever
        self.raptor_retriever = create_raptor_retriever(
            corpus_id=corpus_id,
            k=settings.fusion_k,  # Retrieve more summaries
            level=None,  # Search all levels
            expand_to_chunks=True  # Expand to chunks
        )

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.default_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )

        # Synthesis chain
        prompt = get_synthesis_prompt()
        self.synthesis_chain = prompt | self.llm | StrOutputParser()

        logger.info("RAPTORAgent initialized", corpus_id=corpus_id)

    def build_graph(self) -> CompiledGraph:
        """
        Build the LangGraph workflow.

        Nodes:
        - retrieve: RAPTOR hierarchical retrieval with expansion
        - rerank: Cross-encoder reranking
        - synthesize: LLM synthesis with hierarchy context

        Returns:
            Compiled LangGraph workflow
        """
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("rerank", self._rerank_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Define edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "synthesize")
        workflow.add_edge("synthesize", END)

        logger.info("RAPTORAgent graph built")

        return workflow.compile()

    async def _retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieve documents using RAPTOR hierarchical summaries.

        Args:
            state: Current agent state

        Returns:
            Updated state with retrieved_docs
        """
        query = state["query"]
        start_time = time.time()

        logger.info("RAPTOR retrieve node started", query_preview=query[:100])

        try:
            # RAPTOR retrieval (searches summaries, expands to chunks)
            docs = await self.raptor_retriever.aget_relevant_documents(query)

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                "RAPTOR retrieve node completed",
                results=len(docs),
                duration_ms=round(duration_ms, 2)
            )

            return {
                "retrieved_docs": docs,
            }

        except Exception as e:
            logger.error("RAPTOR retrieve node failed", error=str(e))
            raise

    async def _rerank_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Rerank documents using cross-encoder.

        Args:
            state: Current agent state

        Returns:
            Updated state with reranked docs
        """
        query = state["query"]
        docs = state["retrieved_docs"]
        start_time = time.time()

        logger.info("Rerank node started", num_docs=len(docs))

        try:
            # Rerank
            reranked_docs = await rerank_documents(
                query=query,
                documents=docs,
                top_k=settings.citations_count
            )

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                "Rerank node completed",
                reranked_docs=len(reranked_docs),
                duration_ms=round(duration_ms, 2)
            )

            return {
                "reranked_docs": reranked_docs,
            }

        except Exception as e:
            logger.error("Rerank node failed", error=str(e))
            raise

    async def _synthesize_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Synthesize answer using LLM with hierarchy-aware context.

        Args:
            state: Current agent state

        Returns:
            Updated state with answer and citations
        """
        query = state["query"]
        docs = state["reranked_docs"]
        start_time = time.time()

        logger.info("Synthesize node started", query_preview=query[:100])

        try:
            # Format context with hierarchy information
            context = self._format_hierarchical_context(docs)

            # Generate answer
            answer = await self.synthesis_chain.ainvoke({
                "query": query,
                "context": context
            })

            # Extract citations
            citations = extract_citations_from_answer(answer, docs)

            # Confidence based on chunk weights and citations
            avg_weight = sum(d.metadata.get("weight", 1.0) for d in docs) / len(docs) if docs else 0
            confidence = min(0.9, 0.6 + (avg_weight * 0.3))  # Higher confidence with RAPTOR

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                "Synthesize node completed",
                answer_length=len(answer),
                citations=len(citations),
                confidence=confidence,
                duration_ms=round(duration_ms, 2)
            )

            return {
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
            }

        except Exception as e:
            logger.error("Synthesize node failed", error=str(e))
            raise

    def _format_hierarchical_context(self, docs: list) -> str:
        """
        Format context with hierarchy information.

        Args:
            docs: Retrieved documents

        Returns:
            Formatted context string
        """
        formatted_parts = []

        for i, doc in enumerate(docs, start=1):
            content = doc.page_content
            metadata = doc.metadata

            # Add hierarchy information if available
            source_info = f"Source: {metadata.get('filename', 'Unknown')}"
            if metadata.get('title'):
                source_info += f" - {metadata['title']}"

            # Add RAPTOR weight if available
            weight = metadata.get('weight')
            if weight and weight > 1.0:
                source_info += f" (Importance: {weight:.2f})"

            formatted_parts.append(f"[{i}] {source_info}\n{content}")

        return "\n\n".join(formatted_parts)


def create_raptor_agent(corpus_id: str) -> RAPTORAgent:
    """
    Create a RAPTORAgent instance.

    Args:
        corpus_id: Corpus ID for RAPTOR retrieval

    Returns:
        RAPTORAgent
    """
    return RAPTORAgent(corpus_id=corpus_id)
