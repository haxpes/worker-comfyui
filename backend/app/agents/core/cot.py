"""
CoT Agent: Chain-of-Thought reasoning with step-by-step evidence.
Each reasoning step is linked to supporting sources.
"""
import json
from typing import Dict, Any, Optional, List
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

DECOMPOSITION_PROMPT = """You are breaking down a question into reasoning steps.

Question: {query}

Task: Decompose this into clear, logical reasoning steps. Output JSON:
{{
  "reasoning_chain": [
    {{
      "step": 1,
      "thought": "what to think about",
      "evidence_needed": "what evidence is required",
      "reasoning_type": "deductive|inductive|abductive|analogical"
    }}
  ],
  "final_goal": "what the answer should achieve"
}}

Decomposition:"""

COT_STEP_PROMPT = """You are reasoning through a specific step with evidence.

Question: {query}

Current Step ({step_num}/{total_steps}):
Thought: {thought}
Evidence Needed: {evidence_needed}
Reasoning Type: {reasoning_type}

Retrieved Evidence:
{evidence}

Previous Steps:
{previous_chain}

Task: Reason through this step. Output JSON:
{{
  "reasoning": "detailed step-by-step reasoning",
  "conclusion": "conclusion from this step",
  "confidence": 0.0-1.0,
  "evidence_used": ["evidence_id1", "evidence_id2"],
  "assumptions": ["assumption1"],
  "next_step_hint": "what should come next"
}}

Step Reasoning:"""

FINAL_COT_PROMPT = """You are synthesizing a chain-of-thought reasoning process.

Question: {query}

Complete Reasoning Chain:
{complete_chain}

All Evidence Used:
{all_evidence}

Task: Provide final answer that:
1. Shows the logical flow from steps
2. Cites specific evidence for each claim
3. Acknowledges assumptions and confidence levels
4. Provides clear, well-structured answer

Answer:"""


class CoTAgent(BaseAgent):
    """
    Chain-of-Thought agent with explicit step-by-step reasoning.
    """

    def __init__(
        self,
        corpus_id: Optional[str] = None,
        max_steps: int = 5,
        min_step_confidence: float = 0.6,
        retrieval_k: int = 10
    ):
        """
        Initialize CoTAgent.

        Args:
            corpus_id: Optional corpus ID
            max_steps: Maximum reasoning steps
            min_step_confidence: Minimum confidence per step
            retrieval_k: Documents per retrieval
        """
        super().__init__(name="cot")
        self.corpus_id = corpus_id
        self.max_steps = max_steps
        self.min_step_confidence = min_step_confidence
        self.retrieval_k = retrieval_k

        # Retriever
        self.retriever = create_hybrid_retriever(
            corpus_id=corpus_id,
            bm25_k=retrieval_k,
            dense_k=retrieval_k
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
            "CoTAgent initialized",
            corpus_id=corpus_id,
            max_steps=max_steps
        )

    def build_graph(self) -> CompiledGraph:
        """
        Build the agent's LangGraph workflow.

        Returns:
            Compiled graph
        """
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("decompose", self._decompose_node)
        workflow.add_node("reason_step", self._reason_step_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Define edges
        workflow.set_entry_point("decompose")
        workflow.add_edge("decompose", "reason_step")

        # Conditional: continue reasoning or synthesize
        workflow.add_conditional_edges(
            "reason_step",
            self._should_continue_reasoning,
            {
                "continue": "reason_step",
                "synthesize": "synthesize"
            }
        )

        workflow.add_edge("synthesize", END)

        return workflow.compile()

    async def _decompose_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Decompose query into reasoning steps.

        Args:
            state: Current state

        Returns:
            State updates
        """
        query = state["query"]

        logger.info("Decomposing into reasoning chain", query_preview=query[:100])

        try:
            prompt = ChatPromptTemplate.from_template(DECOMPOSITION_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            result = await chain.ainvoke({"query": query})

            # Parse decomposition
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            decomposition = json.loads(result)

            reasoning_chain = decomposition.get("reasoning_chain", [])[:self.max_steps]

            logger.info(
                "Decomposition completed",
                steps=len(reasoning_chain)
            )

            return {
                "reasoning_chain": reasoning_chain,
                "current_step_idx": 0,
                "step_reasonings": [],
                "all_evidence": [],
                "final_goal": decomposition.get("final_goal", "answer the question")
            }

        except Exception as e:
            logger.error("Decomposition failed", error=str(e))
            # Fallback to single step
            return {
                "reasoning_chain": [{
                    "step": 1,
                    "thought": "Answer the question directly",
                    "evidence_needed": "relevant information",
                    "reasoning_type": "deductive"
                }],
                "current_step_idx": 0,
                "step_reasonings": [],
                "all_evidence": [],
                "final_goal": "answer the question"
            }

    async def _reason_step_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Reason through current step with evidence refresh.

        Args:
            state: Current state

        Returns:
            State updates
        """
        query = state["query"]
        reasoning_chain = state.get("reasoning_chain", [])
        current_step_idx = state.get("current_step_idx", 0)
        step_reasonings = state.get("step_reasonings", [])
        all_evidence = state.get("all_evidence", [])

        if current_step_idx >= len(reasoning_chain):
            return {"current_step_idx": current_step_idx}

        current_step = reasoning_chain[current_step_idx]

        logger.info(
            f"Reasoning step {current_step_idx + 1}/{len(reasoning_chain)}",
            thought=current_step.get("thought", "")[:50]
        )

        # Retrieve evidence for this step
        evidence_query = f"{query} {current_step.get('thought', '')} {current_step.get('evidence_needed', '')}"
        try:
            docs = await self.retriever.ainvoke(evidence_query)

            if docs:
                reranked = await self.reranker.arerank(
                    query=evidence_query,
                    documents=docs,
                    top_k=5
                )
                all_evidence.extend(reranked)
            else:
                reranked = []

        except Exception as e:
            logger.warning(f"Evidence retrieval failed for step {current_step_idx + 1}", error=str(e))
            reranked = []

        # Format evidence
        evidence = "\n\n".join([
            f"[{i+1}] {doc.page_content[:300]}"
            for i, doc in enumerate(reranked)
        ])

        # Previous chain
        previous_chain = "\n\n".join([
            f"**Step {i+1}**: {reasoning.get('reasoning', '')[:100]}\n  → {reasoning.get('conclusion', '')}"
            for i, reasoning in enumerate(step_reasonings)
        ])

        # Reason through step
        try:
            prompt = ChatPromptTemplate.from_template(COT_STEP_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            result = await chain.ainvoke({
                "query": query,
                "step_num": current_step_idx + 1,
                "total_steps": len(reasoning_chain),
                "thought": current_step.get("thought", ""),
                "evidence_needed": current_step.get("evidence_needed", ""),
                "reasoning_type": current_step.get("reasoning_type", "deductive"),
                "evidence": evidence or "No evidence retrieved",
                "previous_chain": previous_chain or "First step"
            })

            # Parse step reasoning
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            step_reasoning = json.loads(result)
            step_reasoning["step_idx"] = current_step_idx
            step_reasonings.append(step_reasoning)

            logger.info(
                f"Step {current_step_idx + 1} reasoning completed",
                confidence=step_reasoning.get("confidence", 0)
            )

        except Exception as e:
            logger.warning(f"Step {current_step_idx + 1} reasoning failed", error=str(e))
            step_reasonings.append({
                "reasoning": f"Step reasoning failed: {str(e)}",
                "conclusion": "Unable to complete this step",
                "confidence": 0.0,
                "evidence_used": [],
                "assumptions": []
            })

        return {
            "current_step_idx": current_step_idx + 1,
            "step_reasonings": step_reasonings,
            "all_evidence": all_evidence
        }

    def _should_continue_reasoning(self, state: AgentState) -> str:
        """
        Decide if reasoning should continue or synthesize.

        Args:
            state: Current state

        Returns:
            Next node name
        """
        current_step_idx = state.get("current_step_idx", 0)
        reasoning_chain = state.get("reasoning_chain", [])

        if current_step_idx >= len(reasoning_chain):
            return "synthesize"
        else:
            return "continue"

    async def _synthesize_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Synthesize final answer from chain-of-thought reasoning.

        Args:
            state: Current state

        Returns:
            State updates with answer
        """
        query = state["query"]
        step_reasonings = state.get("step_reasonings", [])
        all_evidence = state.get("all_evidence", [])
        final_goal = state.get("final_goal", "answer the question")

        logger.info("Synthesizing CoT result", steps=len(step_reasonings))

        # Format complete chain
        complete_chain = "\n\n".join([
            f"**Step {i+1}**\n"
            f"Reasoning: {reasoning.get('reasoning', '')}\n"
            f"Conclusion: {reasoning.get('conclusion', '')}\n"
            f"Confidence: {reasoning.get('confidence', 0):.2f}\n"
            f"Assumptions: {', '.join(reasoning.get('assumptions', []))}"
            for i, reasoning in enumerate(step_reasonings)
        ])

        # Format evidence
        evidence_text = "\n\n".join([
            f"[{i+1}] {doc.page_content[:200]}"
            for i, doc in enumerate(all_evidence[:15])
        ])

        # Synthesize
        try:
            prompt = ChatPromptTemplate.from_template(FINAL_COT_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            answer = await chain.ainvoke({
                "query": query,
                "complete_chain": complete_chain,
                "all_evidence": evidence_text
            })

        except Exception as e:
            logger.error("CoT synthesis failed", error=str(e))
            answer = f"Synthesis failed. Chain of thought:\n\n{complete_chain}"

        # Build citations
        citations = [
            Citation(
                citation_number=i,
                content=doc.page_content[:200],
                doc_id=doc.metadata.get("doc_id", "unknown"),
                metadata={
                    **doc.metadata,
                    "reasoning_step": "multiple"  # Evidence used across steps
                }
            )
            for i, doc in enumerate(all_evidence[:10], 1)
        ]

        # Calculate overall confidence
        confidences = [r.get("confidence", 0.0) for r in step_reasonings]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        logger.info(
            "CoT synthesis completed",
            steps=len(step_reasonings),
            avg_confidence=avg_confidence
        )

        return {
            "answer": answer,
            "citations": [c.dict() for c in citations],
            "confidence": avg_confidence,
            "sources": all_evidence[:10],
            "metadata": {
                "reasoning_model": "chain_of_thought",
                "steps_completed": len(step_reasonings),
                "avg_step_confidence": avg_confidence
            }
        }


def create_cot_agent(corpus_id: Optional[str] = None, **kwargs) -> CoTAgent:
    """
    Create a CoTAgent instance.

    Args:
        corpus_id: Optional corpus ID
        **kwargs: Additional arguments

    Returns:
        CoTAgent
    """
    return CoTAgent(corpus_id=corpus_id, **kwargs)
