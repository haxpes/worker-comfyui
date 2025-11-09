"""
State definitions for LangGraph workflows.
"""
from typing import Annotated, Any, List, Optional, TypedDict
from uuid import UUID
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


# ============================================================================
# AGENT STATE (for individual agent workflows)
# ============================================================================

class AgentState(TypedDict):
    """State for individual agent workflows (subgraphs)."""

    # Query context
    query: str
    refined_query: Optional[str]

    # Retrieval
    retrieved_docs: List[Any]  # List of Document objects
    reranked_docs: List[Any]

    # Synthesis
    answer: str
    citations: List[dict]
    confidence: float

    # Metadata
    agent_name: str
    execution_time_ms: float


# ============================================================================
# MAIN STATE (for router and main graph)
# ============================================================================

class MainState(TypedDict):
    """
    Main state for the entire system workflow.
    Extends MessagesState for conversation history.
    """

    # Conversation history (using LangGraph's add_messages reducer)
    messages: Annotated[List[BaseMessage], add_messages]

    # Query context
    original_query: str
    refined_query: Optional[str]
    query_facets: List[str]

    # Agent routing
    selected_agents: List[str]
    agent_results: dict[str, AgentState]

    # Retrieved evidence (aggregated from all agents)
    retrieved_docs: List[Any]
    seen_doc_ids: set[str]

    # Final synthesis
    answer: str
    citations: List[dict]
    confidence: float

    # Memory context
    thread_id: str
    user_id: Optional[str]
    conversation_summary: Optional[str]

    # Metadata
    execution_time_ms: float
    router_reasoning: Optional[str]


# ============================================================================
# RETRIEVAL RESULT
# ============================================================================

class RetrievalResult(TypedDict):
    """Result from a retrieval operation."""
    docs: List[Any]  # Document objects
    scores: Optional[List[float]]
    retriever_name: str
    execution_time_ms: float


# ============================================================================
# CITATION
# ============================================================================

class Citation(TypedDict):
    """Citation information for an answer."""
    citation_number: int
    content: str
    doc_id: str
    source_title: Optional[str]
    source_type: Optional[str]  # chunk, raptor_summary, entity, etc.
    confidence: float
    metadata: dict


# ============================================================================
# AGENT RESULT
# ============================================================================

class AgentResult(TypedDict):
    """Result from an agent execution."""
    agent_name: str
    answer: str
    citations: List[Citation]
    confidence: float
    retrieved_doc_count: int
    execution_time_ms: float
    metadata: dict


# ============================================================================
# ROUTER DECISION
# ============================================================================

class RouterDecision(TypedDict):
    """Router's agent selection decision."""
    selected_agents: List[str]
    reasoning: str
    confidence: float
    query_analysis: dict  # complexity, intent, domain, etc.


# ============================================================================
# VALIDATION RESULT (Phase 3)
# ============================================================================

class ValidationResult(TypedDict):
    """Result from validator."""
    factuality: float
    consistency: float
    completeness: float
    overall_score: float
    issues: List[str]
    low_confidence_claims: List[str]
    passed: bool  # True if overall_score >= threshold


# ============================================================================
# REFLECTION RESULT (Phase 3 - SelfRAG)
# ============================================================================

class ReflectionResult(TypedDict):
    """Result from reflection in SelfRAG."""
    missing_topics: List[str]
    confidence: float
    needs_more_retrieval: bool
    suggested_searches: List[str]
    iteration_count: int


# ============================================================================
# STATE HELPERS
# ============================================================================

def create_initial_agent_state(query: str, agent_name: str) -> AgentState:
    """
    Create initial agent state.

    Args:
        query: User query
        agent_name: Name of the agent

    Returns:
        Initial AgentState
    """
    return AgentState(
        query=query,
        refined_query=None,
        retrieved_docs=[],
        reranked_docs=[],
        answer="",
        citations=[],
        confidence=0.0,
        agent_name=agent_name,
        execution_time_ms=0.0,
    )


def create_initial_main_state(
    query: str,
    thread_id: str,
    user_id: Optional[str] = None
) -> MainState:
    """
    Create initial main state.

    Args:
        query: User query
        thread_id: Thread/conversation ID
        user_id: Optional user ID

    Returns:
        Initial MainState
    """
    from langchain_core.messages import HumanMessage

    return MainState(
        messages=[HumanMessage(content=query)],
        original_query=query,
        refined_query=None,
        query_facets=[],
        selected_agents=[],
        agent_results={},
        retrieved_docs=[],
        seen_doc_ids=set(),
        answer="",
        citations=[],
        confidence=0.0,
        thread_id=thread_id,
        user_id=user_id,
        conversation_summary=None,
        execution_time_ms=0.0,
        router_reasoning=None,
    )


def merge_agent_results(results: List[AgentResult]) -> dict:
    """
    Merge results from multiple agents.

    Args:
        results: List of AgentResult

    Returns:
        Merged result dictionary
    """
    if not results:
        return {
            "answer": "",
            "citations": [],
            "confidence": 0.0,
        }

    if len(results) == 1:
        return {
            "answer": results[0]["answer"],
            "citations": results[0]["citations"],
            "confidence": results[0]["confidence"],
        }

    # Multi-agent synthesis (simple concatenation for now)
    # TODO: Implement smarter synthesis in Phase 2+
    answers = [r["answer"] for r in results]
    all_citations = []
    citation_num = 1

    for result in results:
        for cite in result["citations"]:
            cite_copy = cite.copy()
            cite_copy["citation_number"] = citation_num
            all_citations.append(cite_copy)
            citation_num += 1

    avg_confidence = sum(r["confidence"] for r in results) / len(results)

    return {
        "answer": "\n\n".join(answers),
        "citations": all_citations,
        "confidence": avg_confidence,
    }
