"""
RefineQuestionAgent: Query enhancement and disambiguation.
Pre-processes queries before routing to main agents.
"""
import json
from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

REFINEMENT_PROMPT = """You are an expert at refining and enhancing search queries.

Original Query: {query}

User Context:
{user_context}

Task: Enhance this query for better retrieval and understanding.

Steps:
1. Disambiguate unclear terms
2. Identify key topical facets
3. Determine search intents
4. Suggest most appropriate agent(s) for this query

Available Agents:
- lean_hybrid: General queries, fast retrieval
- raptor: Long documents, topical queries
- plan_then_read: Multi-part questions
- evidence_first: Compliance, regulatory, citations critical

Output Format (JSON):
{{
  "refined_query": "enhanced query text",
  "facets": ["facet1", "facet2"],
  "search_intents": ["intent1", "intent2"],
  "suggested_agents": ["agent1", "agent2"],
  "reasoning": "why these suggestions"
}}

Refine the query now:"""


class RefineQuestionAgent:
    """
    RefineQuestionAgent enhances queries before routing.
    Not a full agent, but a pre-processing step.
    """

    def __init__(self):
        """Initialize RefineQuestionAgent."""
        self.llm = ChatOpenAI(
            model=settings.default_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )

        logger.info("RefineQuestionAgent initialized")

    async def refine(
        self,
        query: str,
        user_id: str | None = None,
        conversation_summary: str | None = None
    ) -> Dict[str, Any]:
        """
        Refine a query for better retrieval.

        Args:
            query: Original query
            user_id: Optional user ID for context
            conversation_summary: Optional conversation context

        Returns:
            Refinement result dictionary
        """
        logger.info("Refining query", query_preview=query[:100])

        try:
            # Build user context
            user_context = self._build_user_context(
                user_id,
                conversation_summary
            )

            # Refine query
            prompt = ChatPromptTemplate.from_template(REFINEMENT_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            result = await chain.ainvoke({
                "query": query,
                "user_context": user_context
            })

            # Parse JSON
            try:
                if "```json" in result:
                    result = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    result = result.split("```")[1].split("```")[0].strip()

                refinement = json.loads(result)

            except json.JSONDecodeError:
                logger.warning("Failed to parse refinement JSON, using original query")
                refinement = {
                    "refined_query": query,
                    "facets": [],
                    "search_intents": ["general"],
                    "suggested_agents": ["lean_hybrid"],
                    "reasoning": "fallback"
                }

            logger.info(
                "Query refined",
                refined_query=refinement.get("refined_query", "")[:50],
                suggested_agents=refinement.get("suggested_agents", [])
            )

            return refinement

        except Exception as e:
            logger.error("Query refinement failed", error=str(e))
            # Fallback
            return {
                "refined_query": query,
                "facets": [],
                "search_intents": ["general"],
                "suggested_agents": ["lean_hybrid"],
                "reasoning": "error fallback"
            }

    def _build_user_context(
        self,
        user_id: str | None,
        conversation_summary: str | None
    ) -> str:
        """Build user context string."""
        parts = []

        if user_id:
            parts.append(f"User ID: {user_id}")

        if conversation_summary:
            parts.append(f"Conversation Context: {conversation_summary}")

        if not parts:
            parts.append("No prior context available")

        return "\n".join(parts)


def create_refine_question_agent() -> RefineQuestionAgent:
    """
    Create a RefineQuestionAgent instance.

    Returns:
        RefineQuestionAgent
    """
    return RefineQuestionAgent()
