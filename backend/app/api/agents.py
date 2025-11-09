"""
Agents API endpoints with Phase 2 registry integration.
"""
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class AgentInfo(BaseModel):
    """Agent information model."""
    name: str
    display_name: str
    description: str
    category: str
    capabilities: List[str]
    requires_corpus: bool
    supports_streaming: bool
    available: bool = True


class AgentCapabilitiesResponse(BaseModel):
    """Response for agent capabilities query."""
    capability: str
    agents: List[str]


@router.get("/agents", response_model=List[AgentInfo])
async def list_agents(
    req: Request,
    category: str | None = None,
    capability: str | None = None
):
    """
    List all available agents from the registry.

    Args:
        req: FastAPI request object
        category: Optional filter by category (core, extended, utility)
        capability: Optional filter by capability

    Returns:
        List of agent information
    """
    try:
        registry = req.app.state.agent_registry

        # Get agents with optional filters
        agents_metadata = registry.list_agents(
            category=category,
            capability=capability
        )

        # Convert to response format
        agents = [
            AgentInfo(
                name=metadata.name,
                display_name=metadata.display_name,
                description=metadata.description,
                category=metadata.category,
                capabilities=metadata.capabilities,
                requires_corpus=metadata.requires_corpus,
                supports_streaming=metadata.supports_streaming,
                available=True
            )
            for metadata in agents_metadata
        ]

        logger.info(
            "Agents listed",
            total=len(agents),
            category=category,
            capability=capability
        )

        return agents

    except Exception as e:
        logger.error("Failed to list agents", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/capabilities")
async def get_all_capabilities(req: Request) -> Dict[str, List[str]]:
    """
    Get all capabilities and which agents provide them.

    Args:
        req: FastAPI request object

    Returns:
        Dictionary mapping capability to list of agent names
    """
    try:
        registry = req.app.state.agent_registry
        capabilities = registry.get_capabilities()

        logger.info("Capabilities retrieved", total=len(capabilities))

        return capabilities

    except Exception as e:
        logger.error("Failed to get capabilities", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/find/{capability}")
async def find_agents_by_capability(
    capability: str,
    req: Request,
    match_all: bool = False
) -> List[str]:
    """
    Find agents that have a specific capability.

    Args:
        capability: Capability to search for
        req: FastAPI request object
        match_all: If True, require all capabilities (for comma-separated list)

    Returns:
        List of agent names
    """
    try:
        registry = req.app.state.agent_registry

        # Parse comma-separated capabilities
        capabilities = [c.strip() for c in capability.split(",")]

        agents = registry.find_agents_by_capability(
            capabilities=capabilities,
            match_all=match_all
        )

        logger.info(
            "Agents found",
            capabilities=capabilities,
            match_all=match_all,
            found=len(agents)
        )

        return agents

    except Exception as e:
        logger.error("Failed to find agents", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_name}", response_model=AgentInfo)
async def get_agent_info(agent_name: str, req: Request):
    """
    Get information about a specific agent.

    Args:
        agent_name: Agent name
        req: FastAPI request object

    Returns:
        Agent information
    """
    try:
        registry = req.app.state.agent_registry
        metadata = registry.get_metadata(agent_name)

        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_name}' not found"
            )

        return AgentInfo(
            name=metadata.name,
            display_name=metadata.display_name,
            description=metadata.description,
            category=metadata.category,
            capabilities=metadata.capabilities,
            requires_corpus=metadata.requires_corpus,
            supports_streaming=metadata.supports_streaming,
            available=True
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get agent info", agent=agent_name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
