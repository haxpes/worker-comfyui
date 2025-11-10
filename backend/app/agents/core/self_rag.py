"""
SelfRAGAgent: Self-reflective agent with iterative retrieval.
Detects gaps and refines retrieval strategy across iterations.
"""
import json
from typing import Dict, Any, Optional, List, Set
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from app.config import settings
from app.core.state import AgentState, Citation
from app.agents.base import BaseAgent
from app.retrieval.hybrid import create_hybrid_retriever
from app.retrieval.reranker import get_reranker
from app.utils.logger import get_logger

logger = get_logger(__name__)

REFLECTION_PROMPT = """You are a critical reviewer analyzing whether retrieved documents sufficiently answer a question.

Question: {query}

Current Answer Draft: {draft_answer}

Retrieved Documents:
{documents}

Task: Identify gaps in coverage. Output JSON:
{{
  "is_sufficient": true/false,
  "confidence": 0.0-1.0,
  "missing_aspects": ["aspect1", "aspect2"],
  "suggested_refinements": ["refinement1", "refinement2"]
}}

Analysis:"""

SYNTHESIS_PROMPT = """You are an AI assistant providing comprehensive answers through iterative refinement.

Question: {query}

Retrieval Iterations: {iteration_count}

Retrieved Context (across all iterations):
{context}

Coverage Analysis:
{coverage_analysis}

Task: Synthesize a comprehensive answer using ALL retrieved information.
Acknowledge:
1. What is definitively answered
2. What remains partially answered or uncertain
3. Confidence level in the response

Answer:"""


class SelfRAGAgent(BaseAgent):
    """
    Agent with self-reflective iterative retrieval.
    Refines search strategy based on gap detection.
    """

    def __init__(
        self,
        corpus_id: Optional[str] = None,
        max_iterations: int = 3,
        convergence_threshold: float = 0.85,
        k_per_iteration: int = 10
    ):
        """
        Initialize SelfRAGAgent.

        Args:
            corpus_id: Optional corpus ID
            max_iterations: Maximum retrieval iterations
            convergence_threshold: Confidence threshold to stop iteration
            k_per_iteration: Documents per iteration
        """
        super().__init__(name="self_rag")
        self.corpus_id = corpus_id
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.k_per_iteration = k_per_iteration

        # Retriever
        self.retriever = create_hybrid_retriever(
            corpus_id=corpus_id,
            bm25_k=k_per_iteration,
            dense_k=k_per_iteration
        )

        self.reranker = get_reranker()

        # LLM
        self.llm = ChatOpenAI(
            model=settings.default_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )

        self._graph = self.build_graph()

        logger.info(
            "SelfRAGAgent initialized",
            corpus_id=corpus_id,
            max_iterations=max_iterations
        )

    def build_graph(self) -> CompiledGraph:
        """
        Build the agent's LangGraph workflow.

        Returns:
            Compiled graph
        """
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("reflect", self._reflect_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Define edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "retrieve")
        workflow.add_edge("retrieve", "reflect")

        # Conditional: continue or synthesize
        workflow.add_conditional_edges(
            "reflect",
            self._should_continue,
            {
                "continue": "retrieve",
                "synthesize": "synthesize"
            }
        )

        workflow.add_edge("synthesize", END)

        return workflow.compile()

    async def _initialize_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Initialize iteration state.

        Args:
            state: Current state

        Returns:
            State updates
        """
        return {
            "iteration": 0,
            "all_docs": [],
            "seen_doc_ids": set(),
            "is_sufficient": False,
            "confidence": 0.0,
            "missing_aspects": [],
            "draft_answer": ""
        }

    async def _retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieve documents (adaptive based on iteration).

        Args:
            state: Current state

        Returns:
            State updates
        """
        query = state["query"]
        iteration = state.get("iteration", 0)
        seen_doc_ids: Set = state.get("seen_doc_ids", set())
        missing_aspects = state.get("missing_aspects", [])

        # Build retrieval query (refine based on missing aspects)
        if iteration > 0 and missing_aspects:
            # Focus on missing aspects
            refined_query = f"{query} {' '.join(missing_aspects)}"
            logger.info(
                f"Iteration {iteration + 1}: Refined query",
                missing_aspects=missing_aspects
            )
        else:
            refined_query = query

        logger.info(f"Iteration {iteration + 1}: Retrieving")

        # Retrieve
        try:
            docs = await self.retriever.ainvoke(refined_query)

            # Filter out seen documents (novelty filtering)
            new_docs = []
            for doc in docs:
                doc_id = doc.metadata.get("chunk_id", id(doc))
                if doc_id not in seen_doc_ids:
                    new_docs.append(doc)
                    seen_doc_ids.add(doc_id)

            logger.info(f"Retrieved {len(new_docs)} new documents")

            # Rerank new docs
            if new_docs:
                reranked_docs = await self.reranker.arerank(
                    query=refined_query,
                    documents=new_docs,
                    top_k=self.k_per_iteration // 2
                )
            else:
                reranked_docs = []

            # Add to all_docs
            all_docs = state.get("all_docs", [])
            all_docs.extend(reranked_docs)

            return {
                "iteration": iteration + 1,
                "all_docs": all_docs,
                "seen_doc_ids": seen_doc_ids,
                "new_docs": reranked_docs
            }

        except Exception as e:
            logger.error("Retrieval failed", iteration=iteration + 1, error=str(e))
            return {
                "iteration": iteration + 1,
                "new_docs": []
            }

    async def _reflect_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Reflect on retrieval quality and detect gaps.

        Args:
            state: Current state

        Returns:
            State updates
        """
        query = state["query"]
        all_docs = state.get("all_docs", [])
        iteration = state.get("iteration", 0)

        logger.info(f"Iteration {iteration}: Reflecting on {len(all_docs)} total docs")

        if not all_docs:
            return {
                "is_sufficient": False,
                "confidence": 0.0,
                "missing_aspects": ["more information needed"],
                "draft_answer": "No relevant documents found yet."
            }

        # Generate draft answer
        context = "\n\n".join([
            f"[{i+1}] {doc.page_content[:300]}"
            for i, doc in enumerate(all_docs[:10])
        ])

        # Quick draft synthesis
        draft_answer = await self._generate_draft(query, context)

        # Reflection: analyze gaps
        try:
            prompt = ChatPromptTemplate.from_template(REFLECTION_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            reflection_result = await chain.ainvoke({
                "query": query,
                "draft_answer": draft_answer,
                "documents": context
            })

            # Parse JSON
            if "```json" in reflection_result:
                reflection_result = reflection_result.split("```json")[1].split("```")[0].strip()
            elif "```" in reflection_result:
                reflection_result = reflection_result.split("```")[1].split("```")[0].strip()

            reflection = json.loads(reflection_result)

            logger.info(
                f"Reflection complete",
                is_sufficient=reflection.get("is_sufficient", False),
                confidence=reflection.get("confidence", 0.0),
                missing=reflection.get("missing_aspects", [])
            )

            return {
                "is_sufficient": reflection.get("is_sufficient", False),
                "confidence": reflection.get("confidence", 0.0),
                "missing_aspects": reflection.get("missing_aspects", []),
                "draft_answer": draft_answer
            }

        except Exception as e:
            logger.warning("Reflection parsing failed", error=str(e))
            # Fallback: assume sufficient if we have docs
            return {
                "is_sufficient": len(all_docs) >= 5,
                "confidence": 0.7,
                "missing_aspects": [],
                "draft_answer": draft_answer
            }

    def _should_continue(self, state: AgentState) -> str:
        """
        Determine if more iterations are needed.

        Args:
            state: Current state

        Returns:
            Next node name
        """
        iteration = state.get("iteration", 0)
        is_sufficient = state.get("is_sufficient", False)
        confidence = state.get("confidence", 0.0)

        # Stop conditions
        if iteration >= self.max_iterations:
            logger.info(f"Max iterations ({self.max_iterations}) reached")
            return "synthesize"

        if is_sufficient and confidence >= self.convergence_threshold:
            logger.info(f"Convergence achieved (confidence={confidence})")
            return "synthesize"

        logger.info(f"Continuing to iteration {iteration + 1}")
        return "continue"

    async def _synthesize_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Final synthesis from all retrieved documents.

        Args:
            state: Current state

        Returns:
            State updates with answer
        """
        query = state["query"]
        all_docs = state.get("all_docs", [])
        iteration = state.get("iteration", 0)
        confidence = state.get("confidence", 0.0)

        if not all_docs:
            return {
                "answer": "I couldn't find sufficient information to answer your question comprehensively.",
                "citations": [],
                "confidence": 0.0,
                "metadata": {"iterations": iteration}
            }

        # Build comprehensive context
        context = "\n\n".join([
            f"[{i+1}] {doc.page_content}"
            for i, doc in enumerate(all_docs)
        ])

        # Coverage analysis
        coverage_analysis = f"Retrieved {len(all_docs)} documents across {iteration} iterations. Confidence: {confidence:.2f}"

        # Generate final answer
        prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)
        chain = prompt | self.llm | StrOutputParser()

        answer = await chain.ainvoke({
            "query": query,
            "iteration_count": iteration,
            "context": context,
            "coverage_analysis": coverage_analysis
        })

        # Build citations
        citations = [
            Citation(
                citation_number=i,
                content=doc.page_content[:200],
                doc_id=doc.metadata.get("doc_id", "unknown"),
                metadata=doc.metadata
            )
            for i, doc in enumerate(all_docs, 1)
        ]

        logger.info(
            "Final synthesis completed",
            iterations=iteration,
            docs=len(all_docs),
            confidence=confidence
        )

        return {
            "answer": answer,
            "citations": [c.dict() for c in citations],
            "confidence": min(confidence, 0.95),  # Cap at 0.95
            "sources": all_docs,
            "metadata": {
                "iterations": iteration,
                "total_docs": len(all_docs),
                "self_assessed_confidence": confidence
            }
        }

    async def _generate_draft(self, query: str, context: str) -> str:
        """
        Generate quick draft answer for reflection.

        Args:
            query: Query
            context: Context

        Returns:
            Draft answer
        """
        try:
            prompt = f"Question: {query}\n\nContext: {context}\n\nBrief answer:"
            result = await self.llm.ainvoke(prompt)
            return result.content.strip()
        except:
            return "Unable to generate draft"


def create_self_rag_agent(corpus_id: Optional[str] = None, **kwargs) -> SelfRAGAgent:
    """
    Create a SelfRAGAgent instance.

    Args:
        corpus_id: Optional corpus ID
        **kwargs: Additional arguments

    Returns:
        SelfRAGAgent
    """
    return SelfRAGAgent(corpus_id=corpus_id, **kwargs)
