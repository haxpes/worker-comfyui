"""
Agents API endpoints.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()


class AgentInfo(BaseModel):
    """Agent information model."""
    name: str
    description: str
    available: bool
    phase: int  # Which phase this agent is implemented in


@router.get("/agents", response_model=List[AgentInfo])
async def list_agents():
    """
    List all available agents.

    Returns:
        List of agent information
    """
    agents = [
        AgentInfo(
            name="lean_hybrid",
            description="Fast hybrid retrieval agent for general queries (BM25 + Dense + RRF + MMR + Cross-encoder)",
            available=True,
            phase=1
        ),
        AgentInfo(
            name="raptor",
            description="RAPTOR agent for long documents with hierarchical summaries",
            available=False,
            phase=2
        ),
        AgentInfo(
            name="plan_then_read",
            description="Multi-part question decomposition agent",
            available=False,
            phase=2
        ),
        AgentInfo(
            name="evidence_first",
            description="Compliance-focused agent with strict citation mapping",
            available=False,
            phase=2
        ),
        AgentInfo(
            name="graph_on_demand",
            description="Relationship-aware agent using ephemeral knowledge graphs",
            available=False,
            phase=3
        ),
        AgentInfo(
            name="self_rag",
            description="Self-reflective agent with iterative retrieval",
            available=False,
            phase=3
        ),
        AgentInfo(
            name="distill_first",
            description="Fast agent using pre-distilled knowledge",
            available=False,
            phase=4
        ),
        AgentInfo(
            name="hrm",
            description="Hierarchical reasoning model for complex decisions",
            available=False,
            phase=4
        ),
        AgentInfo(
            name="cot",
            description="Chain-of-thought agent for step-by-step reasoning",
            available=False,
            phase=4
        ),
        AgentInfo(
            name="react",
            description="Tool-using agent with reasoning and acting loops",
            available=False,
            phase=4
        ),
    ]

    return agents


@router.get("/agents/{agent_name}")
async def get_agent_info(agent_name: str):
    """
    Get information about a specific agent.

    Args:
        agent_name: Agent name

    Returns:
        Agent information
    """
    agents_dict = {
        "lean_hybrid": AgentInfo(
            name="lean_hybrid",
            description="Fast hybrid retrieval agent for general queries (BM25 + Dense + RRF + MMR + Cross-encoder)",
            available=True,
            phase=1
        ),
    }

    agent = agents_dict.get(agent_name)

    if not agent:
        return {
            "error": "Agent not found",
            "available_agents": list(agents_dict.keys())
        }

    return agent
