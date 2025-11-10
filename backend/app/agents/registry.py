"""
Agent Registry: Centralized management of all AI-SME agents.
Provides agent metadata, initialization, and discovery.
"""
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentMetadata:
    """Metadata for an agent."""
    name: str
    display_name: str
    description: str
    category: str  # "core", "extended", "utility"
    capabilities: List[str]
    requirements: Dict[str, Any] = field(default_factory=dict)
    requires_corpus: bool = False
    supports_streaming: bool = True
    initialization_fn: Optional[Callable] = None


class AgentRegistry:
    """
    Registry for all AI-SME agents.
    Manages agent metadata, initialization, and discovery.
    """

    def __init__(self):
        """Initialize agent registry."""
        self._agents: Dict[str, AgentMetadata] = {}
        self._instances: Dict[str, Any] = {}
        self._register_default_agents()

        logger.info("AgentRegistry initialized")

    def _register_default_agents(self):
        """Register all default agents."""
        from app.agents.core.lean_hybrid import create_lean_hybrid_agent
        from app.agents.core.raptor_agent import create_raptor_agent
        from app.agents.core.plan_then_read import create_plan_then_read_agent
        from app.agents.core.evidence_first import create_evidence_first_agent
        from app.agents.core.graph_on_demand import create_graph_on_demand_agent
        from app.agents.core.self_rag import create_self_rag_agent

        # Phase 1-2 Core agents
        self.register(
            AgentMetadata(
                name="lean_hybrid",
                display_name="Lean Hybrid Agent",
                description="Fast hybrid retrieval with BM25 + dense search, RRF fusion, and cross-encoder reranking",
                category="core",
                capabilities=["general_qa", "fact_finding", "keyword_search", "semantic_search"],
                requires_corpus=False,
                supports_streaming=True,
                initialization_fn=create_lean_hybrid_agent
            )
        )

        self.register(
            AgentMetadata(
                name="raptor",
                display_name="RAPTOR Agent",
                description="Hierarchical retrieval using RAPTOR summaries for complex, multi-document queries",
                category="core",
                capabilities=["complex_qa", "multi_document", "hierarchical_retrieval", "summarization"],
                requires_corpus=True,
                supports_streaming=True,
                initialization_fn=create_raptor_agent
            )
        )

        self.register(
            AgentMetadata(
                name="plan_then_read",
                display_name="Plan-Then-Read Agent",
                description="Decomposes complex queries into sub-questions and synthesizes comprehensive answers",
                category="core",
                capabilities=["complex_qa", "decomposition", "multi_step_reasoning", "comprehensive_answers"],
                requires_corpus=False,
                supports_streaming=True,
                initialization_fn=create_plan_then_read_agent
            )
        )

        self.register(
            AgentMetadata(
                name="evidence_first",
                display_name="Evidence-First Agent",
                description="Extractive QA with strict citation mapping for compliance and regulatory queries",
                category="core",
                capabilities=["extractive_qa", "compliance", "regulatory", "strict_citations", "minimal_hallucination"],
                requires_corpus=False,
                supports_streaming=True,
                initialization_fn=create_evidence_first_agent
            )
        )

        # Phase 3 Core agents
        self.register(
            AgentMetadata(
                name="graph_on_demand",
                display_name="GraphOnDemand Agent",
                description="Relationship-aware retrieval using knowledge graph with entity and community search",
                category="core",
                capabilities=["relationship_qa", "graph_retrieval", "entity_search", "community_analysis"],
                requires_corpus=False,
                supports_streaming=True,
                initialization_fn=create_graph_on_demand_agent
            )
        )

        self.register(
            AgentMetadata(
                name="self_rag",
                display_name="SelfRAG Agent",
                description="Self-reflective agent with iterative retrieval and gap detection for comprehensive answers",
                category="core",
                capabilities=["iterative_retrieval", "self_reflection", "gap_detection", "comprehensive_qa"],
                requires_corpus=False,
                supports_streaming=True,
                initialization_fn=create_self_rag_agent
            )
        )

        logger.info("Default agents registered", count=len(self._agents))

    def register(self, metadata: AgentMetadata):
        """
        Register an agent.

        Args:
            metadata: Agent metadata
        """
        if metadata.name in self._agents:
            logger.warning(f"Agent '{metadata.name}' already registered, overwriting")

        self._agents[metadata.name] = metadata

        logger.debug(
            "Agent registered",
            name=metadata.name,
            category=metadata.category,
            capabilities=metadata.capabilities
        )

    def unregister(self, agent_name: str):
        """
        Unregister an agent.

        Args:
            agent_name: Agent name
        """
        if agent_name in self._agents:
            del self._agents[agent_name]
            logger.debug(f"Agent '{agent_name}' unregistered")

        if agent_name in self._instances:
            del self._instances[agent_name]

    def get_metadata(self, agent_name: str) -> Optional[AgentMetadata]:
        """
        Get agent metadata.

        Args:
            agent_name: Agent name

        Returns:
            Agent metadata or None
        """
        return self._agents.get(agent_name)

    def list_agents(
        self,
        category: Optional[str] = None,
        capability: Optional[str] = None
    ) -> List[AgentMetadata]:
        """
        List all registered agents, optionally filtered.

        Args:
            category: Filter by category
            capability: Filter by capability

        Returns:
            List of agent metadata
        """
        agents = list(self._agents.values())

        if category:
            agents = [a for a in agents if a.category == category]

        if capability:
            agents = [a for a in agents if capability in a.capabilities]

        return agents

    def get_agent_names(self) -> List[str]:
        """
        Get all registered agent names.

        Returns:
            List of agent names
        """
        return list(self._agents.keys())

    def has_agent(self, agent_name: str) -> bool:
        """
        Check if an agent is registered.

        Args:
            agent_name: Agent name

        Returns:
            True if registered
        """
        return agent_name in self._agents

    def initialize_agent(
        self,
        agent_name: str,
        corpus_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Initialize an agent instance.

        Args:
            agent_name: Agent name
            corpus_id: Optional corpus ID (required for some agents)
            **kwargs: Additional initialization parameters

        Returns:
            Agent instance

        Raises:
            ValueError: If agent not found or initialization fails
        """
        metadata = self.get_metadata(agent_name)

        if not metadata:
            raise ValueError(f"Agent '{agent_name}' not found in registry")

        if not metadata.initialization_fn:
            raise ValueError(f"Agent '{agent_name}' has no initialization function")

        # Check requirements
        if metadata.requires_corpus and not corpus_id:
            raise ValueError(f"Agent '{agent_name}' requires corpus_id")

        logger.info(f"Initializing agent '{agent_name}'", corpus_id=corpus_id)

        try:
            # Call initialization function
            if metadata.requires_corpus:
                agent = metadata.initialization_fn(corpus_id=corpus_id, **kwargs)
            else:
                agent = metadata.initialization_fn(corpus_id=corpus_id, **kwargs)

            return agent

        except Exception as e:
            logger.error(f"Failed to initialize agent '{agent_name}'", error=str(e))
            raise

    def initialize_all_agents(
        self,
        corpus_id: Optional[str] = None,
        lazy: bool = True
    ) -> Dict[str, Any]:
        """
        Initialize all registered agents.

        Args:
            corpus_id: Optional default corpus ID
            lazy: If True, only initialize agents that don't require corpus

        Returns:
            Dictionary mapping agent name to agent instance
        """
        logger.info("Initializing all agents", corpus_id=corpus_id, lazy=lazy)

        instances = {}

        for agent_name, metadata in self._agents.items():
            try:
                # Skip corpus-required agents in lazy mode without corpus_id
                if lazy and metadata.requires_corpus and not corpus_id:
                    logger.debug(
                        f"Skipping agent '{agent_name}' (requires corpus)",
                        lazy=lazy
                    )
                    continue

                agent = self.initialize_agent(agent_name, corpus_id=corpus_id)
                instances[agent_name] = agent

            except Exception as e:
                logger.error(
                    f"Failed to initialize agent '{agent_name}'",
                    error=str(e)
                )
                # Continue with other agents

        logger.info(
            "Agent initialization completed",
            initialized=len(instances),
            total=len(self._agents)
        )

        return instances

    def get_or_create_agent(
        self,
        agent_name: str,
        corpus_id: Optional[str] = None,
        cache: bool = True,
        **kwargs
    ) -> Any:
        """
        Get cached agent instance or create new one.

        Args:
            agent_name: Agent name
            corpus_id: Optional corpus ID
            cache: Whether to cache the instance
            **kwargs: Initialization parameters

        Returns:
            Agent instance
        """
        # Check cache first
        cache_key = f"{agent_name}:{corpus_id or 'default'}"

        if cache and cache_key in self._instances:
            logger.debug(f"Returning cached agent '{agent_name}'")
            return self._instances[cache_key]

        # Initialize new agent
        agent = self.initialize_agent(agent_name, corpus_id=corpus_id, **kwargs)

        # Cache if requested
        if cache:
            self._instances[cache_key] = agent

        return agent

    def clear_cache(self):
        """Clear all cached agent instances."""
        self._instances.clear()
        logger.info("Agent cache cleared")

    def get_agent_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about all agents.

        Returns:
            Dictionary mapping agent name to info dict
        """
        info = {}

        for name, metadata in self._agents.items():
            info[name] = {
                "name": metadata.name,
                "display_name": metadata.display_name,
                "description": metadata.description,
                "category": metadata.category,
                "capabilities": metadata.capabilities,
                "requires_corpus": metadata.requires_corpus,
                "supports_streaming": metadata.supports_streaming,
                "requirements": metadata.requirements
            }

        return info

    def find_agents_by_capability(
        self,
        capabilities: List[str],
        match_all: bool = False
    ) -> List[str]:
        """
        Find agents that have specific capabilities.

        Args:
            capabilities: List of required capabilities
            match_all: If True, agent must have all capabilities

        Returns:
            List of agent names
        """
        matching_agents = []

        for name, metadata in self._agents.items():
            agent_caps = set(metadata.capabilities)
            required_caps = set(capabilities)

            if match_all:
                # Must have all capabilities
                if required_caps.issubset(agent_caps):
                    matching_agents.append(name)
            else:
                # Must have at least one capability
                if required_caps.intersection(agent_caps):
                    matching_agents.append(name)

        return matching_agents

    def get_capabilities(self) -> Dict[str, List[str]]:
        """
        Get all capabilities and which agents provide them.

        Returns:
            Dictionary mapping capability to list of agent names
        """
        capabilities: Dict[str, List[str]] = {}

        for name, metadata in self._agents.items():
            for cap in metadata.capabilities:
                if cap not in capabilities:
                    capabilities[cap] = []
                capabilities[cap].append(name)

        return capabilities


# Global registry instance
_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    """
    Get global agent registry instance.

    Returns:
        AgentRegistry
    """
    global _registry

    if _registry is None:
        _registry = AgentRegistry()

    return _registry


def register_agent(metadata: AgentMetadata):
    """
    Register an agent in the global registry.

    Args:
        metadata: Agent metadata
    """
    registry = get_agent_registry()
    registry.register(metadata)


def get_agent(
    agent_name: str,
    corpus_id: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Get or create an agent from the global registry.

    Args:
        agent_name: Agent name
        corpus_id: Optional corpus ID
        **kwargs: Initialization parameters

    Returns:
        Agent instance
    """
    registry = get_agent_registry()
    return registry.get_or_create_agent(agent_name, corpus_id=corpus_id, **kwargs)
