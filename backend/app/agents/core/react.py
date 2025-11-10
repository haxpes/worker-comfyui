"""
ReAct Agent: Reasoning + Acting with tool usage loops.
Alternates between reasoning about what to do and taking actions.
"""
import json
from typing import Dict, Any, Optional, List, Callable
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
from app.retrieval.entity import create_entity_retriever
from app.utils.logger import get_logger

logger = get_logger(__name__)

REACT_PROMPT = """You are a ReAct agent that alternates between Reasoning and Acting.

Question: {query}

Previous Steps:
{previous_steps}

Available Tools:
- search_corpus: Search the document corpus for information
- search_entities: Search for named entities and their relationships
- calculate: Perform calculations
- finish: Complete the task with final answer

Task: Decide next action. Output JSON:
{{
  "thought": "reasoning about what to do next",
  "action": "tool_name",
  "action_input": "input for the tool",
  "is_final": false
}}

OR if ready to answer:
{{
  "thought": "final reasoning",
  "action": "finish",
  "action_input": "final answer text",
  "is_final": true
}}

ReAct:"""

SYNTHESIS_PROMPT = """You are synthesizing results from a ReAct reasoning loop.

Question: {query}

Complete Action Trace:
{action_trace}

All Observations:
{all_observations}

Task: Provide final comprehensive answer based on the reasoning and action loop.

Answer:"""


class ReActAgent(BaseAgent):
    """
    ReAct (Reasoning + Acting) agent with tool usage.
    """

    def __init__(
        self,
        corpus_id: Optional[str] = None,
        max_iterations: int = 5,
        retrieval_k: int = 10
    ):
        """
        Initialize ReActAgent.

        Args:
            corpus_id: Optional corpus ID
            max_iterations: Maximum reasoning-action iterations
            retrieval_k: Documents per retrieval
        """
        super().__init__(name="react")
        self.corpus_id = corpus_id
        self.max_iterations = max_iterations
        self.retrieval_k = retrieval_k

        # Retrievers (tools)
        self.corpus_retriever = create_hybrid_retriever(
            corpus_id=corpus_id,
            bm25_k=retrieval_k,
            dense_k=retrieval_k
        )

        self.entity_retriever = create_entity_retriever(
            corpus_id=corpus_id,
            k=retrieval_k
        )

        # LLM
        self.llm = ChatOpenAI(
            model=settings.default_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )

        # Tool registry
        self.tools = {
            "search_corpus": self._search_corpus_tool,
            "search_entities": self._search_entities_tool,
            "calculate": self._calculate_tool,
            "finish": self._finish_tool
        }

        self._graph = self.build_graph()

        logger.info(
            "ReActAgent initialized",
            corpus_id=corpus_id,
            max_iterations=max_iterations,
            tools=list(self.tools.keys())
        )

    def build_graph(self) -> CompiledGraph:
        """
        Build the agent's LangGraph workflow.

        Returns:
            Compiled graph
        """
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("act", self._act_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Define edges
        workflow.set_entry_point("reason")
        workflow.add_edge("reason", "act")

        # Conditional: continue loop or finish
        workflow.add_conditional_edges(
            "act",
            self._should_continue,
            {
                "continue": "reason",
                "finish": "synthesize"
            }
        )

        workflow.add_edge("synthesize", END)

        return workflow.compile()

    async def _reason_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Reasoning step: Decide what action to take.

        Args:
            state: Current state

        Returns:
            State updates
        """
        query = state["query"]
        iteration = state.get("react_iteration", 0)
        action_trace = state.get("action_trace", [])

        logger.info(f"ReAct iteration {iteration + 1}: Reasoning")

        # Format previous steps
        previous_steps = "\n\n".join([
            f"**Step {i+1}**\n"
            f"Thought: {step.get('thought', '')}\n"
            f"Action: {step.get('action', '')}({step.get('action_input', '')})\n"
            f"Observation: {step.get('observation', '')[:200]}"
            for i, step in enumerate(action_trace)
        ])

        try:
            prompt = ChatPromptTemplate.from_template(REACT_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            result = await chain.ainvoke({
                "query": query,
                "previous_steps": previous_steps or "No previous steps"
            })

            # Parse reasoning
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            reasoning = json.loads(result)

            logger.info(
                f"Reasoning completed",
                action=reasoning.get("action", ""),
                is_final=reasoning.get("is_final", False)
            )

            return {
                "current_reasoning": reasoning,
                "react_iteration": iteration
            }

        except Exception as e:
            logger.error("Reasoning failed", error=str(e))
            # Fallback to finish
            return {
                "current_reasoning": {
                    "thought": "Error in reasoning, finishing",
                    "action": "finish",
                    "action_input": "Unable to complete reasoning",
                    "is_final": True
                },
                "react_iteration": iteration
            }

    async def _act_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Acting step: Execute the chosen action.

        Args:
            state: Current state

        Returns:
            State updates
        """
        reasoning = state.get("current_reasoning", {})
        action = reasoning.get("action", "finish")
        action_input = reasoning.get("action_input", "")
        action_trace = state.get("action_trace", [])
        iteration = state.get("react_iteration", 0)

        logger.info(f"ReAct iteration {iteration + 1}: Acting - {action}")

        # Execute tool
        tool_fn = self.tools.get(action)

        if tool_fn:
            try:
                observation = await tool_fn(action_input)
            except Exception as e:
                logger.error(f"Tool '{action}' execution failed", error=str(e))
                observation = f"Error: {str(e)}"
        else:
            observation = f"Unknown tool: {action}"

        # Record action
        action_trace.append({
            "thought": reasoning.get("thought", ""),
            "action": action,
            "action_input": action_input,
            "observation": observation,
            "is_final": reasoning.get("is_final", False)
        })

        logger.info(
            f"Action completed",
            observation_length=len(str(observation)) if observation else 0
        )

        return {
            "action_trace": action_trace,
            "react_iteration": iteration + 1,
            "last_observation": observation
        }

    def _should_continue(self, state: AgentState) -> str:
        """
        Decide if ReAct loop should continue or finish.

        Args:
            state: Current state

        Returns:
            Next node name
        """
        action_trace = state.get("action_trace", [])
        iteration = state.get("react_iteration", 0)

        if not action_trace:
            return "continue"

        last_action = action_trace[-1]

        # Check if final
        if last_action.get("is_final", False) or last_action.get("action") == "finish":
            return "finish"

        # Check max iterations
        if iteration >= self.max_iterations:
            logger.info(f"Max iterations ({self.max_iterations}) reached")
            return "finish"

        return "continue"

    async def _synthesize_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Synthesize final answer from ReAct trace.

        Args:
            state: Current state

        Returns:
            State updates with answer
        """
        query = state["query"]
        action_trace = state.get("action_trace", [])

        logger.info("Synthesizing ReAct result", iterations=len(action_trace))

        # Check if finish action has answer
        if action_trace and action_trace[-1].get("action") == "finish":
            final_answer = action_trace[-1].get("action_input", "")

            # Use finish answer directly if substantial
            if len(final_answer) > 50:
                return {
                    "answer": final_answer,
                    "citations": [],
                    "confidence": 0.8,
                    "metadata": {
                        "reasoning_model": "react",
                        "iterations": len(action_trace)
                    }
                }

        # Format action trace
        action_trace_text = "\n\n".join([
            f"**Iteration {i+1}**\n"
            f"Thought: {step.get('thought', '')}\n"
            f"Action: {step.get('action', '')}({step.get('action_input', '')})\n"
            f"Observation: {step.get('observation', '')[:300]}"
            for i, step in enumerate(action_trace)
        ])

        # All observations
        observations = "\n\n".join([
            f"[{i+1}] {step.get('observation', '')[:500]}"
            for i, step in enumerate(action_trace)
            if step.get('observation')
        ])

        # Synthesize
        try:
            prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            answer = await chain.ainvoke({
                "query": query,
                "action_trace": action_trace_text,
                "all_observations": observations
            })

        except Exception as e:
            logger.error("ReAct synthesis failed", error=str(e))
            answer = f"Synthesis failed. Action trace:\n\n{action_trace_text}"

        logger.info("ReAct synthesis completed", iterations=len(action_trace))

        return {
            "answer": answer,
            "citations": [],
            "confidence": 0.75,
            "metadata": {
                "reasoning_model": "react",
                "iterations": len(action_trace),
                "tools_used": list(set(step.get("action") for step in action_trace))
            }
        }

    # Tool implementations

    async def _search_corpus_tool(self, query: str) -> str:
        """Search corpus for information."""
        try:
            docs = await self.corpus_retriever.ainvoke(query)

            if docs:
                results = "\n\n".join([
                    f"[{i+1}] {doc.page_content[:300]}"
                    for i, doc in enumerate(docs[:5])
                ])
                return f"Found {len(docs)} documents:\n\n{results}"
            else:
                return "No documents found"

        except Exception as e:
            return f"Search error: {str(e)}"

    async def _search_entities_tool(self, query: str) -> str:
        """Search for entities."""
        try:
            docs = await self.entity_retriever.ainvoke(query)

            if docs:
                entities = set()
                for doc in docs[:5]:
                    entity_name = doc.metadata.get("entity_name", "Unknown")
                    entity_type = doc.metadata.get("entity_type", "Unknown")
                    entities.add(f"{entity_name} ({entity_type})")

                results = "Found entities: " + ", ".join(entities)
                return results
            else:
                return "No entities found"

        except Exception as e:
            return f"Entity search error: {str(e)}"

    async def _calculate_tool(self, expression: str) -> str:
        """Perform calculation."""
        try:
            # Simple safe eval for basic math
            # In production, use a proper math parser
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Calculation result: {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"

    async def _finish_tool(self, answer: str) -> str:
        """Finish with answer."""
        return answer


def create_react_agent(corpus_id: Optional[str] = None, **kwargs) -> ReActAgent:
    """
    Create a ReActAgent instance.

    Args:
        corpus_id: Optional corpus ID
        **kwargs: Additional arguments

    Returns:
        ReActAgent
    """
    return ReActAgent(corpus_id=corpus_id, **kwargs)
