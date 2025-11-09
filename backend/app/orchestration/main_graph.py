"""
Main orchestration graph: Central AI-SME workflow.
Integrates query refinement, routing, agent execution, and critique.
"""
from typing import Any, Dict, List, Optional, TypedDict
from uuid import uuid4
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from langchain_core.documents import Document
from app.config import settings
from app.core.state import AgentResult, Citation
from app.core.memory import get_memory_manager
from app.agents.extended.refine_question import create_refine_question_agent
from app.agents.extended.critique import create_critique_agent
from app.orchestration.router import create_router
from app.orchestration.synthesis import merge_agent_results, deduplicate_citations, rank_citations_by_relevance
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MainGraphState(TypedDict):
    """State for main orchestration graph."""
    # Input
    query: str
    user_id: Optional[str]
    corpus_id: Optional[str]
    thread_id: str

    # Configuration
    enable_query_refinement: bool
    enable_critique: bool
    multi_agent: bool  # If True, run multiple agents and merge results

    # Query refinement
    refined_query: Optional[str]
    query_facets: List[str]
    suggested_agents: List[str]

    # Routing
    selected_agent: str
    routing_confidence: float
    routing_reasoning: str
    query_analysis: Dict[str, Any]

    # Agent execution
    agent_results: List[AgentResult]  # Support multiple agents

    # Critique
    critique: Optional[Dict[str, Any]]
    was_improved: bool

    # Final output
    answer: str
    citations: List[Citation]
    confidence: float
    metadata: Dict[str, Any]
    sources: List[Document]  # For critique


class MainGraph:
    """
    Main orchestration graph for AI-SME system.
    Coordinates query refinement, routing, agent execution, and critique.
    """

    def __init__(
        self,
        agent_registry: Dict[str, Any],
        enable_query_refinement: bool = True,
        enable_critique: bool = True,
        critique_threshold: float = 0.7
    ):
        """
        Initialize main graph.

        Args:
            agent_registry: Dictionary mapping agent names to agent instances
            enable_query_refinement: Whether to use RefineQuestionAgent
            enable_critique: Whether to use CritiqueAgent
            critique_threshold: Quality threshold for critique improvement
        """
        self.agent_registry = agent_registry
        self.enable_query_refinement = enable_query_refinement
        self.enable_critique = enable_critique
        self.critique_threshold = critique_threshold

        # Initialize components
        self.refine_agent = create_refine_question_agent() if enable_query_refinement else None
        self.critique_agent = create_critique_agent() if enable_critique else None
        self.router = create_router()
        self.memory = get_memory_manager()

        # Build graph
        self.graph = self.build_graph()

        logger.info(
            "MainGraph initialized",
            agents=list(agent_registry.keys()),
            query_refinement=enable_query_refinement,
            critique=enable_critique
        )

    def build_graph(self) -> CompiledGraph:
        """
        Build the main orchestration graph.

        Returns:
            Compiled LangGraph
        """
        workflow = StateGraph(MainGraphState)

        # Add nodes
        workflow.add_node("refine_query", self._refine_query_node)
        workflow.add_node("route", self._route_node)
        workflow.add_node("execute_agent", self._execute_agent_node)
        workflow.add_node("critique", self._critique_node)
        workflow.add_node("finalize", self._finalize_node)

        # Define edges
        workflow.set_entry_point("refine_query")
        workflow.add_edge("refine_query", "route")
        workflow.add_edge("route", "execute_agent")

        # Conditional edge after agent execution
        workflow.add_conditional_edges(
            "execute_agent",
            self._should_critique,
            {
                "critique": "critique",
                "finalize": "finalize"
            }
        )

        workflow.add_edge("critique", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile(
            checkpointer=self.memory.checkpointer
        )

    async def _refine_query_node(self, state: MainGraphState) -> Dict[str, Any]:
        """
        Refine the query using RefineQuestionAgent.

        Args:
            state: Current state

        Returns:
            State updates
        """
        query = state["query"]

        # Skip if disabled
        if not state.get("enable_query_refinement", self.enable_query_refinement):
            logger.debug("Query refinement disabled")
            return {
                "refined_query": query,
                "query_facets": [],
                "suggested_agents": []
            }

        logger.info("Refining query", query_preview=query[:100])

        try:
            result = await self.refine_agent.refine(query)

            refined = result.get("refined_query", query)
            facets = result.get("facets", [])
            suggested = result.get("suggested_agents", [])

            logger.info(
                "Query refined",
                original_length=len(query),
                refined_length=len(refined),
                facets=len(facets),
                suggested_agents=suggested
            )

            return {
                "refined_query": refined,
                "query_facets": facets,
                "suggested_agents": suggested
            }

        except Exception as e:
            logger.error("Query refinement failed", error=str(e))
            return {
                "refined_query": query,
                "query_facets": [],
                "suggested_agents": []
            }

    async def _route_node(self, state: MainGraphState) -> Dict[str, Any]:
        """
        Route to the best agent using Router.

        Args:
            state: Current state

        Returns:
            State updates
        """
        query = state["query"]
        refined_query = state.get("refined_query")
        user_id = state.get("user_id")
        suggested_agents = state.get("suggested_agents", [])

        logger.info("Routing query", query_preview=query[:100])

        try:
            # Build context for routing
            context = {}
            if suggested_agents:
                context["suggested_agents"] = suggested_agents

            # Route
            decision = await self.router.route(
                query=query,
                refined_query=refined_query,
                user_id=user_id,
                context=context
            )

            selected = decision["selected_agent"]
            confidence = decision["confidence"]
            reasoning = decision["reasoning"]
            analysis = decision.get("query_analysis", {})

            logger.info(
                "Routing completed",
                selected_agent=selected,
                confidence=confidence,
                reasoning=reasoning[:100]
            )

            return {
                "selected_agent": selected,
                "routing_confidence": confidence,
                "routing_reasoning": reasoning,
                "query_analysis": analysis
            }

        except Exception as e:
            logger.error("Routing failed", error=str(e))
            # Fallback to lean_hybrid
            return {
                "selected_agent": "lean_hybrid",
                "routing_confidence": 0.5,
                "routing_reasoning": f"Fallback due to error: {str(e)}",
                "query_analysis": {}
            }

    async def _execute_agent_node(self, state: MainGraphState) -> Dict[str, Any]:
        """
        Execute the selected agent(s).

        Args:
            state: Current state

        Returns:
            State updates
        """
        query = state.get("refined_query") or state["query"]
        selected_agent = state["selected_agent"]
        corpus_id = state.get("corpus_id")
        multi_agent = state.get("multi_agent", False)

        logger.info(
            "Executing agent",
            agent=selected_agent,
            query_preview=query[:100],
            multi_agent=multi_agent
        )

        try:
            agents_to_run = []

            if multi_agent:
                # Run multiple agents and merge
                # Use routing confidence to decide which agents to run
                agents_to_run = [selected_agent]

                # Add complementary agents based on query analysis
                query_type = state.get("query_analysis", {}).get("type", "")
                if query_type == "complex":
                    agents_to_run.append("plan_then_read")

                logger.info("Multi-agent execution", agents=agents_to_run)
            else:
                agents_to_run = [selected_agent]

            # Execute agents
            results = []
            sources_combined = []

            for agent_name in agents_to_run:
                agent = self.agent_registry.get(agent_name)

                if not agent:
                    logger.warning(f"Agent '{agent_name}' not found in registry")
                    continue

                # Run agent
                agent_result = await agent.run(
                    query=query,
                    corpus_id=corpus_id,
                    stream=False  # No streaming in orchestration
                )

                # Extract result
                if "answer" in agent_result:
                    # Final state
                    result_dict = {
                        "agent_name": agent_name,
                        "answer": agent_result["answer"],
                        "citations": agent_result.get("citations", []),
                        "confidence": agent_result.get("confidence", 0.8),
                        "metadata": agent_result.get("metadata", {})
                    }
                    results.append(result_dict)

                    # Collect sources for critique
                    if "sources" in agent_result:
                        sources_combined.extend(agent_result["sources"])

            logger.info(
                "Agent execution completed",
                num_results=len(results),
                agents=[r["agent_name"] for r in results]
            )

            return {
                "agent_results": results,
                "sources": sources_combined
            }

        except Exception as e:
            logger.error("Agent execution failed", error=str(e))
            # Return error result
            return {
                "agent_results": [{
                    "agent_name": selected_agent,
                    "answer": f"Agent execution failed: {str(e)}",
                    "citations": [],
                    "confidence": 0.0,
                    "metadata": {"error": str(e)}
                }],
                "sources": []
            }

    def _should_critique(self, state: MainGraphState) -> str:
        """
        Determine if critique should be applied.

        Args:
            state: Current state

        Returns:
            Next node name
        """
        if not state.get("enable_critique", self.enable_critique):
            return "finalize"

        # Check if we have valid results
        results = state.get("agent_results", [])
        if not results:
            return "finalize"

        # Check confidence - only critique if confidence is uncertain
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        if avg_confidence < 0.75:
            return "critique"

        return "finalize"

    async def _critique_node(self, state: MainGraphState) -> Dict[str, Any]:
        """
        Critique and potentially improve the answer.

        Args:
            state: Current state

        Returns:
            State updates
        """
        query = state.get("refined_query") or state["query"]
        results = state["agent_results"]
        sources = state.get("sources", [])

        logger.info("Critiquing answer", num_results=len(results))

        try:
            # Get the best answer for critique
            best_result = max(results, key=lambda r: r["confidence"])
            answer = best_result["answer"]

            # Critique and improve
            critique_result = await self.critique_agent.critique_and_improve(
                query=query,
                answer=answer,
                sources=sources[:10],  # Limit for token budget
                improvement_threshold=self.critique_threshold
            )

            was_improved = critique_result["was_improved"]
            improved_answer = critique_result["improved_answer"]
            critique = critique_result["critique"]

            # Update the best result
            if was_improved:
                best_result["answer"] = improved_answer
                best_result["metadata"]["improved"] = True
                best_result["metadata"]["original_quality"] = critique.get("overall_quality", 0)

            logger.info(
                "Critique completed",
                was_improved=was_improved,
                quality=critique.get("overall_quality", 0)
            )

            return {
                "agent_results": results,
                "critique": critique,
                "was_improved": was_improved
            }

        except Exception as e:
            logger.error("Critique failed", error=str(e))
            return {
                "critique": None,
                "was_improved": False
            }

    async def _finalize_node(self, state: MainGraphState) -> Dict[str, Any]:
        """
        Finalize the response by merging results if needed.

        Args:
            state: Current state

        Returns:
            State updates with final answer
        """
        results = state["agent_results"]
        multi_agent = state.get("multi_agent", False)

        logger.info("Finalizing response", num_results=len(results))

        try:
            # Merge results if multiple agents
            if len(results) > 1 and multi_agent:
                strategy = "combine"  # Could be configurable
                final = merge_agent_results(results, strategy=strategy)
            else:
                # Single agent result
                final = {
                    "answer": results[0]["answer"],
                    "citations": results[0]["citations"],
                    "confidence": results[0]["confidence"],
                    "metadata": results[0]["metadata"]
                }

            # Deduplicate and rank citations
            citations = deduplicate_citations(final["citations"])
            citations = rank_citations_by_relevance(citations, max_citations=10)

            # Add orchestration metadata
            metadata = final["metadata"]
            metadata["orchestration"] = {
                "selected_agent": state["selected_agent"],
                "routing_confidence": state["routing_confidence"],
                "query_refined": state.get("refined_query") is not None,
                "was_critiqued": state.get("critique") is not None,
                "was_improved": state.get("was_improved", False),
                "num_agents": len(results)
            }

            logger.info(
                "Response finalized",
                answer_length=len(final["answer"]),
                citations=len(citations),
                confidence=final["confidence"]
            )

            return {
                "answer": final["answer"],
                "citations": citations,
                "confidence": final["confidence"],
                "metadata": metadata
            }

        except Exception as e:
            logger.error("Finalization failed", error=str(e))
            # Fallback to first result
            return {
                "answer": results[0]["answer"] if results else "Error: No results",
                "citations": results[0]["citations"] if results else [],
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }

    async def run(
        self,
        query: str,
        user_id: Optional[str] = None,
        corpus_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        enable_query_refinement: Optional[bool] = None,
        enable_critique: Optional[bool] = None,
        multi_agent: bool = False
    ) -> Dict[str, Any]:
        """
        Run the main orchestration graph.

        Args:
            query: User query
            user_id: Optional user ID
            corpus_id: Optional corpus ID
            thread_id: Optional thread ID for memory
            enable_query_refinement: Override default setting
            enable_critique: Override default setting
            multi_agent: Whether to run multiple agents

        Returns:
            Final result dictionary
        """
        # Generate thread_id if not provided
        if not thread_id:
            thread_id = f"thread_{uuid4()}"

        # Initial state
        initial_state = {
            "query": query,
            "user_id": user_id,
            "corpus_id": corpus_id,
            "thread_id": thread_id,
            "enable_query_refinement": enable_query_refinement if enable_query_refinement is not None else self.enable_query_refinement,
            "enable_critique": enable_critique if enable_critique is not None else self.enable_critique,
            "multi_agent": multi_agent,
            "refined_query": None,
            "query_facets": [],
            "suggested_agents": [],
            "selected_agent": "",
            "routing_confidence": 0.0,
            "routing_reasoning": "",
            "query_analysis": {},
            "agent_results": [],
            "critique": None,
            "was_improved": False,
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "metadata": {},
            "sources": []
        }

        logger.info(
            "Starting main graph execution",
            query_preview=query[:100],
            thread_id=thread_id,
            multi_agent=multi_agent
        )

        try:
            # Run graph
            config = {"configurable": {"thread_id": thread_id}}

            final_state = await self.graph.ainvoke(initial_state, config=config)

            # Update query patterns for learning
            if user_id and final_state.get("selected_agent"):
                await self.memory.update_query_patterns(
                    user_id=user_id,
                    query=query,
                    selected_agents=[final_state["selected_agent"]],
                    success=final_state.get("confidence", 0) > 0.7
                )

            logger.info(
                "Main graph execution completed",
                confidence=final_state.get("confidence", 0),
                thread_id=thread_id
            )

            return final_state

        except Exception as e:
            logger.error("Main graph execution failed", error=str(e), thread_id=thread_id)
            raise


def create_main_graph(
    agent_registry: Dict[str, Any],
    enable_query_refinement: bool = True,
    enable_critique: bool = True,
    critique_threshold: float = 0.7
) -> MainGraph:
    """
    Create a MainGraph instance.

    Args:
        agent_registry: Dictionary of agent name -> agent instance
        enable_query_refinement: Whether to use RefineQuestionAgent
        enable_critique: Whether to use CritiqueAgent
        critique_threshold: Quality threshold for improvement

    Returns:
        MainGraph instance
    """
    return MainGraph(
        agent_registry=agent_registry,
        enable_query_refinement=enable_query_refinement,
        enable_critique=enable_critique,
        critique_threshold=critique_threshold
    )
