"""
Multi-agent synthesis: Combines results from multiple agents.
"""
from typing import List, Dict, Any
from langchain_core.documents import Document
from app.core.state import AgentResult, Citation
from app.utils.logger import get_logger

logger = get_logger(__name__)


def merge_agent_results(
    results: List[AgentResult],
    strategy: str = "best"
) -> Dict[str, Any]:
    """
    Merge results from multiple agents.

    Args:
        results: List of AgentResult dictionaries
        strategy: Merging strategy ("best", "combine", "vote")

    Returns:
        Merged result dictionary
    """
    if not results:
        return {
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "metadata": {"error": "no results"}
        }

    if len(results) == 1:
        # Single result, return as-is
        return {
            "answer": results[0]["answer"],
            "citations": results[0]["citations"],
            "confidence": results[0]["confidence"],
            "metadata": {
                "agent": results[0]["agent_name"],
                "single_agent": True
            }
        }

    logger.info(
        "Merging agent results",
        num_results=len(results),
        strategy=strategy
    )

    if strategy == "best":
        return _merge_best(results)
    elif strategy == "combine":
        return _merge_combine(results)
    elif strategy == "vote":
        return _merge_vote(results)
    else:
        logger.warning(f"Unknown strategy '{strategy}', using 'best'")
        return _merge_best(results)


def _merge_best(results: List[AgentResult]) -> Dict[str, Any]:
    """
    Select the best result based on confidence.

    Args:
        results: List of agent results

    Returns:
        Best result
    """
    # Sort by confidence
    sorted_results = sorted(
        results,
        key=lambda r: r.get("confidence", 0.0),
        reverse=True
    )

    best = sorted_results[0]

    logger.info(
        "Selected best result",
        agent=best["agent_name"],
        confidence=best["confidence"]
    )

    return {
        "answer": best["answer"],
        "citations": best["citations"],
        "confidence": best["confidence"],
        "metadata": {
            "strategy": "best",
            "selected_agent": best["agent_name"],
            "all_agents": [r["agent_name"] for r in results],
            "all_confidences": [r["confidence"] for r in results]
        }
    }


def _merge_combine(results: List[AgentResult]) -> Dict[str, Any]:
    """
    Combine all results into a comprehensive answer.

    Args:
        results: List of agent results

    Returns:
        Combined result
    """
    # Combine answers with agent attribution
    combined_answers = []
    for result in results:
        agent_name = result["agent_name"]
        answer = result["answer"]
        combined_answers.append(f"**{agent_name}**: {answer}")

    final_answer = "\n\n".join(combined_answers)

    # Merge citations, renumbering to avoid conflicts
    all_citations = []
    citation_num = 1

    for result in results:
        for citation in result["citations"]:
            # Renumber citation
            new_citation = citation.copy()
            new_citation["citation_number"] = citation_num
            new_citation["source_agent"] = result["agent_name"]
            all_citations.append(new_citation)
            citation_num += 1

    # Average confidence
    avg_confidence = sum(r["confidence"] for r in results) / len(results)

    logger.info(
        "Combined results",
        num_agents=len(results),
        total_citations=len(all_citations),
        avg_confidence=avg_confidence
    )

    return {
        "answer": final_answer,
        "citations": all_citations,
        "confidence": avg_confidence,
        "metadata": {
            "strategy": "combine",
            "agents": [r["agent_name"] for r in results],
            "individual_confidences": [r["confidence"] for r in results]
        }
    }


def _merge_vote(results: List[AgentResult]) -> Dict[str, Any]:
    """
    Use voting to select best answer segments.
    Simple implementation: use highest confidence.

    Args:
        results: List of agent results

    Returns:
        Voted result
    """
    # For now, this is similar to "best" but could be enhanced
    # with more sophisticated voting mechanisms
    return _merge_best(results)


def deduplicate_citations(citations: List[Citation]) -> List[Citation]:
    """
    Remove duplicate citations based on document ID.

    Args:
        citations: List of citations

    Returns:
        Deduplicated citations
    """
    seen_doc_ids = set()
    unique_citations = []

    for citation in citations:
        doc_id = citation.get("doc_id")
        if doc_id and doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            unique_citations.append(citation)
        elif not doc_id:
            # No doc_id, keep it
            unique_citations.append(citation)

    if len(unique_citations) < len(citations):
        logger.debug(
            "Deduplicated citations",
            original=len(citations),
            unique=len(unique_citations)
        )

    return unique_citations


def rank_citations_by_relevance(
    citations: List[Citation],
    max_citations: int = 5
) -> List[Citation]:
    """
    Rank citations by relevance and return top K.

    Args:
        citations: List of citations
        max_citations: Maximum citations to return

    Returns:
        Top K citations
    """
    # Sort by confidence/relevance
    sorted_citations = sorted(
        citations,
        key=lambda c: c.get("confidence", 0.5),
        reverse=True
    )

    return sorted_citations[:max_citations]
