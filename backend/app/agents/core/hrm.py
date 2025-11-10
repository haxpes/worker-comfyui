"""
HRM Agent: Hierarchical Reasoning Model with strategy/execution alternation.
For complex decisions requiring multi-level reasoning.
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

STRATEGY_PROMPT = """You are a strategic planner analyzing a complex question.

Question: {query}

Current Context:
{context}

Task: Develop a hierarchical reasoning strategy. Output JSON:
{{
  "strategy_level": "high|mid|low",
  "reasoning_steps": [
    {{
      "step": 1,
      "description": "what to analyze",
      "type": "information_gathering|analysis|synthesis|decision"
    }}
  ],
  "success_criteria": ["criterion1", "criterion2"],
  "expected_complexity": "low|medium|high"
}}

Strategy:"""

EXECUTION_PROMPT = """You are executing a reasoning step.

Question: {query}

Current Step: {step_description}
Step Type: {step_type}

Retrieved Evidence:
{evidence}

Previous Steps Results:
{previous_results}

Task: Execute this reasoning step. Provide:
1. Analysis for this step
2. Key findings
3. Whether this step is complete

Output JSON:
{{
  "analysis": "detailed analysis",
  "findings": ["finding1", "finding2"],
  "complete": true/false,
  "needs_more_evidence": true/false
}}

Execution:"""

CONVERGENCE_PROMPT = """You are evaluating whether hierarchical reasoning has converged.

Question: {query}

Strategy: {strategy}

Executed Steps:
{executed_steps}

Task: Determine if reasoning is complete. Output JSON:
{{
  "converged": true/false,
  "confidence": 0.0-1.0,
  "remaining_steps": ["step1", "step2"],
  "ready_for_synthesis": true/false
}}

Evaluation:"""

SYNTHESIS_PROMPT = """You are synthesizing results from hierarchical reasoning.

Question: {query}

Strategy: {strategy}

All Step Results:
{all_results}

Decision Matrix:
{decision_matrix}

Task: Provide final comprehensive answer incorporating all reasoning levels.

Answer:"""


class HRMAgent(BaseAgent):
    """
    Hierarchical Reasoning Model agent with strategy/execution alternation.
    """

    def __init__(
        self,
        corpus_id: Optional[str] = None,
        max_steps: int = 5,
        convergence_threshold: float = 0.85,
        retrieval_k: int = 10
    ):
        """
        Initialize HRMAgent.

        Args:
            corpus_id: Optional corpus ID
            max_steps: Maximum reasoning steps
            convergence_threshold: Confidence threshold for convergence
            retrieval_k: Documents per retrieval
        """
        super().__init__(name="hrm")
        self.corpus_id = corpus_id
        self.max_steps = max_steps
        self.convergence_threshold = convergence_threshold
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
            "HRMAgent initialized",
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
        workflow.add_node("formulate_strategy", self._formulate_strategy_node)
        workflow.add_node("execute_step", self._execute_step_node)
        workflow.add_node("check_convergence", self._check_convergence_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Define edges
        workflow.set_entry_point("formulate_strategy")
        workflow.add_edge("formulate_strategy", "execute_step")

        # Conditional: continue executing or check convergence
        workflow.add_conditional_edges(
            "execute_step",
            self._should_check_convergence,
            {
                "continue": "execute_step",
                "check": "check_convergence"
            }
        )

        # Conditional: converged or continue
        workflow.add_conditional_edges(
            "check_convergence",
            self._has_converged,
            {
                "synthesize": "synthesize",
                "continue": "execute_step"
            }
        )

        workflow.add_edge("synthesize", END)

        return workflow.compile()

    async def _formulate_strategy_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Formulate hierarchical reasoning strategy.

        Args:
            state: Current state

        Returns:
            State updates
        """
        query = state["query"]

        logger.info("Formulating strategy", query_preview=query[:100])

        try:
            # Initial retrieval for context
            docs = await self.retriever.ainvoke(query)
            context = "\n\n".join([doc.page_content[:200] for doc in docs[:3]])

            # Formulate strategy
            prompt = ChatPromptTemplate.from_template(STRATEGY_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            result = await chain.ainvoke({
                "query": query,
                "context": context
            })

            # Parse strategy
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            strategy = json.loads(result)

            logger.info(
                "Strategy formulated",
                steps=len(strategy.get("reasoning_steps", [])),
                complexity=strategy.get("expected_complexity", "unknown")
            )

            return {
                "strategy": strategy,
                "current_step_idx": 0,
                "step_results": [],
                "all_evidence": docs
            }

        except Exception as e:
            logger.error("Strategy formulation failed", error=str(e))
            # Fallback strategy
            return {
                "strategy": {
                    "strategy_level": "mid",
                    "reasoning_steps": [
                        {"step": 1, "description": "Analyze question", "type": "analysis"}
                    ],
                    "success_criteria": ["answer provided"],
                    "expected_complexity": "medium"
                },
                "current_step_idx": 0,
                "step_results": [],
                "all_evidence": []
            }

    async def _execute_step_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute current reasoning step.

        Args:
            state: Current state

        Returns:
            State updates
        """
        query = state["query"]
        strategy = state.get("strategy", {})
        current_step_idx = state.get("current_step_idx", 0)
        step_results = state.get("step_results", [])
        all_evidence = state.get("all_evidence", [])

        steps = strategy.get("reasoning_steps", [])

        if current_step_idx >= len(steps):
            # No more steps
            return {"current_step_idx": current_step_idx}

        current_step = steps[current_step_idx]

        logger.info(
            f"Executing step {current_step_idx + 1}/{len(steps)}",
            description=current_step.get("description", "")
        )

        # Retrieve evidence for this step if needed
        if current_step.get("type") == "information_gathering":
            step_query = f"{query} {current_step.get('description', '')}"
            new_docs = await self.retriever.ainvoke(step_query)
            all_evidence.extend(new_docs[:5])

        # Build evidence context
        evidence = "\n\n".join([
            f"[{i+1}] {doc.page_content[:300]}"
            for i, doc in enumerate(all_evidence[-10:])  # Last 10 docs
        ])

        # Previous results
        previous_results = "\n\n".join([
            f"Step {i+1}: {result.get('analysis', '')}[:200]"
            for i, result in enumerate(step_results)
        ])

        # Execute step
        try:
            prompt = ChatPromptTemplate.from_template(EXECUTION_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            result = await chain.ainvoke({
                "query": query,
                "step_description": current_step.get("description", ""),
                "step_type": current_step.get("type", "analysis"),
                "evidence": evidence,
                "previous_results": previous_results or "No previous steps"
            })

            # Parse result
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            step_result = json.loads(result)
            step_results.append(step_result)

            logger.info(
                f"Step {current_step_idx + 1} completed",
                complete=step_result.get("complete", False)
            )

            return {
                "current_step_idx": current_step_idx + 1,
                "step_results": step_results,
                "all_evidence": all_evidence
            }

        except Exception as e:
            logger.warning(f"Step {current_step_idx + 1} execution failed", error=str(e))
            # Add placeholder result
            step_results.append({
                "analysis": "Step execution failed",
                "findings": [],
                "complete": False
            })

            return {
                "current_step_idx": current_step_idx + 1,
                "step_results": step_results,
                "all_evidence": all_evidence
            }

    def _should_check_convergence(self, state: AgentState) -> str:
        """
        Decide if we should check convergence.

        Args:
            state: Current state

        Returns:
            Next node name
        """
        current_step_idx = state.get("current_step_idx", 0)
        strategy = state.get("strategy", {})
        steps = strategy.get("reasoning_steps", [])

        # Check every 2 steps or when all steps done
        if current_step_idx >= len(steps) or current_step_idx % 2 == 0:
            return "check"
        else:
            return "continue"

    async def _check_convergence_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Check if reasoning has converged.

        Args:
            state: Current state

        Returns:
            State updates
        """
        query = state["query"]
        strategy = state.get("strategy", {})
        step_results = state.get("step_results", [])
        current_step_idx = state.get("current_step_idx", 0)

        logger.info("Checking convergence")

        # Format executed steps
        executed_steps = "\n\n".join([
            f"Step {i+1}: {result.get('analysis', '')[:100]}"
            for i, result in enumerate(step_results)
        ])

        try:
            prompt = ChatPromptTemplate.from_template(CONVERGENCE_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            result = await chain.ainvoke({
                "query": query,
                "strategy": json.dumps(strategy, indent=2),
                "executed_steps": executed_steps
            })

            # Parse convergence
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            convergence = json.loads(result)

            logger.info(
                "Convergence check",
                converged=convergence.get("converged", False),
                confidence=convergence.get("confidence", 0)
            )

            return {
                "converged": convergence.get("converged", False),
                "convergence_confidence": convergence.get("confidence", 0.0),
                "ready_for_synthesis": convergence.get("ready_for_synthesis", False)
            }

        except Exception as e:
            logger.warning("Convergence check failed", error=str(e))
            # Assume converged if all steps done
            steps = strategy.get("reasoning_steps", [])
            return {
                "converged": current_step_idx >= len(steps),
                "convergence_confidence": 0.7,
                "ready_for_synthesis": True
            }

    def _has_converged(self, state: AgentState) -> str:
        """
        Determine if converged or should continue.

        Args:
            state: Current state

        Returns:
            Next node name
        """
        converged = state.get("converged", False)
        confidence = state.get("convergence_confidence", 0.0)
        current_step_idx = state.get("current_step_idx", 0)

        if (converged and confidence >= self.convergence_threshold) or current_step_idx >= self.max_steps:
            return "synthesize"
        else:
            return "continue"

    async def _synthesize_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Synthesize final answer from hierarchical reasoning.

        Args:
            state: Current state

        Returns:
            State updates with answer
        """
        query = state["query"]
        strategy = state.get("strategy", {})
        step_results = state.get("step_results", [])
        all_evidence = state.get("all_evidence", [])
        confidence = state.get("convergence_confidence", 0.8)

        logger.info("Synthesizing HRM result", steps=len(step_results))

        # Format all results
        all_results = "\n\n".join([
            f"**Step {i+1}**\n{result.get('analysis', '')}"
            for i, result in enumerate(step_results)
        ])

        # Build decision matrix
        decision_matrix = "\n".join([
            f"- {i+1}. {', '.join(result.get('findings', []))}"
            for i, result in enumerate(step_results)
        ])

        # Synthesize
        prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)
        chain = prompt | self.llm | StrOutputParser()

        answer = await chain.ainvoke({
            "query": query,
            "strategy": json.dumps(strategy, indent=2),
            "all_results": all_results,
            "decision_matrix": decision_matrix
        })

        # Build citations
        citations = [
            Citation(
                citation_number=i,
                content=doc.page_content[:200],
                doc_id=doc.metadata.get("doc_id", "unknown"),
                metadata=doc.metadata
            )
            for i, doc in enumerate(all_evidence[:10], 1)
        ]

        logger.info(
            "HRM synthesis completed",
            steps=len(step_results),
            confidence=confidence
        )

        return {
            "answer": answer,
            "citations": [c.dict() for c in citations],
            "confidence": confidence,
            "sources": all_evidence[:10],
            "metadata": {
                "reasoning_model": "hierarchical",
                "steps_executed": len(step_results),
                "strategy_level": strategy.get("strategy_level", "mid")
            }
        }


def create_hrm_agent(corpus_id: Optional[str] = None, **kwargs) -> HRMAgent:
    """
    Create an HRMAgent instance.

    Args:
        corpus_id: Optional corpus ID
        **kwargs: Additional arguments

    Returns:
        HRMAgent
    """
    return HRMAgent(corpus_id=corpus_id, **kwargs)
