"""
LeanHybridAgent: Simple, fast hybrid retrieval agent.
Uses BM25 + Dense → RRF + MMR → Cross-encoder rerank → LLM synthesis.
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
from app.retrieval.bm25 import create_bm25_retriever
from app.retrieval.dense import create_dense_retriever
from app.retrieval.fusion import apply_fusion_and_diversity
from app.retrieval.reranker import rerank_documents
from app.utils.prompts import get_synthesis_prompt, format_context_with_citations, extract_citations_from_answer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LeanHybridAgent(BaseAgent):
    """
    LeanHybridAgent for general queries with hybrid retrieval.

    Workflow:
    1. Retrieve: BM25 + Dense retrievers in parallel
    2. Fuse: RRF fusion + MMR diversity
    3. Rerank: Cross-encoder reranking
    4. Synthesize: LLM generates answer with citations
    """

    def __init__(self):
        """Initialize LeanHybridAgent."""
        super().__init__(name="lean_hybrid")

        # Initialize retrievers
        self.bm25_retriever = create_bm25_retriever()
        self.dense_retriever = create_dense_retriever()

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.default_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )

        # Synthesis chain
        prompt = get_synthesis_prompt()
        self.synthesis_chain = prompt | self.llm | StrOutputParser()

        logger.info("LeanHybridAgent initialized")

    def build_graph(self) -> CompiledGraph:
        """
        Build the LangGraph workflow.

        Nodes:
        - retrieve: Parallel BM25 + Dense retrieval
        - fuse: RRF fusion + MMR diversity
        - rerank: Cross-encoder reranking
        - synthesize: LLM synthesis with citations

        Returns:
            Compiled LangGraph workflow
        """
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("fuse", self._fuse_node)
        workflow.add_node("rerank", self._rerank_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Define edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "fuse")
        workflow.add_edge("fuse", "rerank")
        workflow.add_edge("rerank", "synthesize")
        workflow.add_edge("synthesize", END)

        logger.info("LeanHybridAgent graph built")

        return workflow.compile()

    async def _retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieve documents using BM25 and Dense retrievers in parallel.

        Args:
            state: Current agent state

        Returns:
            Updated state with retrieved_docs
        """
        query = state["query"]
        start_time = time.time()

        logger.info("Retrieve node started", query_preview=query[:100])

        try:
            # Parallel retrieval
            import asyncio

            bm25_task = self.bm25_retriever.aget_relevant_documents(query)
            dense_task = self.dense_retriever.aget_relevant_documents(query)

            bm25_docs, dense_docs = await asyncio.gather(bm25_task, dense_task)

            # Store both lists for fusion
            retrieved_docs = [bm25_docs, dense_docs]

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                "Retrieve node completed",
                bm25_results=len(bm25_docs),
                dense_results=len(dense_docs),
                duration_ms=round(duration_ms, 2)
            )

            return {
                "retrieved_docs": retrieved_docs,  # List of lists for fusion
            }

        except Exception as e:
            logger.error("Retrieve node failed", error=str(e))
            raise

    async def _fuse_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Apply RRF fusion and MMR diversity.

        Args:
            state: Current agent state

        Returns:
            Updated state with fused docs
        """
        query = state["query"]
        doc_lists = state["retrieved_docs"]
        start_time = time.time()

        logger.info("Fuse node started", num_lists=len(doc_lists))

        try:
            # Apply fusion and diversity
            fused_docs = await apply_fusion_and_diversity(
                doc_lists=doc_lists,
                query=query,
                fusion_k=settings.fusion_k,
                mmr_k=settings.mmr_k,
                lambda_param=settings.mmr_lambda
            )

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                "Fuse node completed",
                fused_docs=len(fused_docs),
                duration_ms=round(duration_ms, 2)
            )

            return {
                "retrieved_docs": fused_docs,  # Now a single list
            }

        except Exception as e:
            logger.error("Fuse node failed", error=str(e))
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
        Synthesize answer using LLM with citations.

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
            # Format context with citations
            context = format_context_with_citations(docs)

            # Generate answer
            answer = await self.synthesis_chain.ainvoke({
                "query": query,
                "context": context
            })

            # Extract citations
            citations = extract_citations_from_answer(answer, docs)

            # Simple confidence (could be improved with validator)
            confidence = 0.8 if citations else 0.5

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


def create_lean_hybrid_agent() -> LeanHybridAgent:
    """
    Create a LeanHybridAgent instance.

    Returns:
        LeanHybridAgent
    """
    return LeanHybridAgent()
