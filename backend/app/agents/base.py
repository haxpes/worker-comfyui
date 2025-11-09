"""
Base agent interface and utilities.
"""
from typing import Protocol, Dict, Any
from abc import ABC, abstractmethod
from langgraph.graph import CompiledGraph
from app.core.state import AgentState


class Agent(Protocol):
    """Protocol for agent implementations."""

    name: str

    async def run(self, query: str) -> AgentState:
        """Run the agent on a query."""
        ...

    def get_graph(self) -> CompiledGraph:
        """Get the compiled LangGraph workflow."""
        ...


class BaseAgent(ABC):
    """
    Base class for all agents.
    Provides common utilities and interface.
    """

    def __init__(self, name: str):
        """
        Initialize base agent.

        Args:
            name: Agent name
        """
        self.name = name
        self._graph: CompiledGraph | None = None

    @abstractmethod
    def build_graph(self) -> CompiledGraph:
        """
        Build and compile the LangGraph workflow.

        Returns:
            Compiled graph
        """
        pass

    def get_graph(self) -> CompiledGraph:
        """
        Get the compiled graph, building if necessary.

        Returns:
            Compiled graph
        """
        if self._graph is None:
            self._graph = self.build_graph()
        return self._graph

    async def run(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Run the agent on a query.

        Args:
            query: User query
            **kwargs: Additional parameters

        Returns:
            Agent result dictionary
        """
        from app.core.state import create_initial_agent_state

        # Create initial state
        initial_state = create_initial_agent_state(query, self.name)

        # Merge any additional kwargs
        initial_state.update(kwargs)

        # Get graph
        graph = self.get_graph()

        # Execute workflow
        result = await graph.ainvoke(initial_state)

        return result
