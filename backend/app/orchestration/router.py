"""
Router (Mixture of Experts): Selects appropriate agents based on query analysis.
"""
import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from app.config import settings
from app.core.memory import get_memory_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)

ROUTING_PROMPT = """You are an expert at analyzing queries and selecting the best AI agent to handle them.

Query: {query}

Query Analysis Context:
{context}

User Query Patterns (learned preferences):
{query_patterns}

Available Agents:
1. **lean_hybrid**: Fast general-purpose retrieval (BM25 + Dense + RRF + MMR)
   - Best for: Simple factual queries, quick lookups, general questions
   - Pros: Fastest, lowest cost
   - Cons: May miss nuance in complex queries

2. **raptor**: Hierarchical topical retrieval
   - Best for: Long documents, topical queries, broad subject exploration
   - Pros: Excellent topical recall, good for summaries
   - Cons: Requires RAPTOR summaries to be built

3. **plan_then_read**: Multi-part query decomposition
   - Best for: Complex multi-part questions, structured analysis
   - Pros: Systematic, thorough
   - Cons: Slower, more expensive

4. **evidence_first**: Extractive QA with strict citations
   - Best for: Compliance, regulatory, audit-critical queries
   - Pros: Highest citation integrity, minimal hallucination risk
   - Cons: Can be terse, requires high-quality sources

5. **graph_on_demand**: Relationship-aware retrieval using knowledge graph
   - Best for: Relationship queries, "how are X and Y related", entity-focused questions
   - Pros: Excellent for understanding connections, community insights
   - Cons: Requires entity extraction and graph building

6. **self_rag**: Self-reflective iterative retrieval
   - Best for: Very complex queries requiring comprehensive coverage, research questions
   - Pros: Most thorough, detects and fills gaps, highest completeness
   - Cons: Slowest, most expensive

7. **distill_first**: Pre-distilled knowledge with corpus fallback
   - Best for: Repeated queries, similar questions asked before
   - Pros: Fastest for repeated queries, learns from conversation history
   - Cons: May be stale, requires conversation history

8. **hrm**: Hierarchical reasoning model (strategy + execution)
   - Best for: Complex decisions, multi-level analysis, strategic planning
   - Pros: Structured reasoning, decision matrix output
   - Cons: Complex, slower than simple agents

9. **cot**: Chain-of-thought reasoning
   - Best for: Queries requiring transparent logic, step-by-step explanations
   - Pros: Shows reasoning process, tracks assumptions
   - Cons: Verbose, may over-explain

10. **react**: Reasoning + Acting with tools
   - Best for: Dynamic problems requiring tool usage, calculations, multi-step actions
   - Pros: Can use tools, flexible problem solving
   - Cons: May require multiple iterations

Task: Select the SINGLE best agent for this query.

Consider:
1. Query complexity (simple vs. complex)
2. Domain (general vs. compliance/regulatory)
3. User intent (quick answer vs. thorough analysis)
4. Citation requirements (nice-to-have vs. critical)
5. User's historical preferences

Output Format (JSON):
{{
  "selected_agent": "agent_name",
  "confidence": 0.0-1.0,
  "reasoning": "detailed explanation of why this agent was selected",
  "query_analysis": {{
    "complexity": "simple|moderate|complex",
    "domain": "general|technical|compliance",
    "intent": "factual|analytical|verification",
    "citation_critical": true|false
  }}
}}

Select the best agent now:"""


class Router:
    """
    Router (Mixture of Experts) for agent selection.
    Analyzes queries and selects the most appropriate agent.
    """

    def __init__(self):
        """Initialize Router."""
        self.llm = ChatOpenAI(
            model=settings.default_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )

        self.memory_manager = get_memory_manager()

        # Agent availability (updated based on what's initialized)
        self.available_agents = [
            "lean_hybrid",
            "raptor",
            "plan_then_read",
            "evidence_first",
            "graph_on_demand",
            "self_rag",
            "distill_first",
            "hrm",
            "cot",
            "react"
        ]

        logger.info("Router initialized", available_agents=self.available_agents)

    async def route(
        self,
        query: str,
        refined_query: str | None = None,
        user_id: str | None = None,
        context: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        Route a query to the appropriate agent.

        Args:
            query: Original query
            refined_query: Refined query (if available)
            user_id: Optional user ID for pattern learning
            context: Additional context

        Returns:
            Routing decision dictionary
        """
        logger.info("Routing query", query_preview=query[:100])

        try:
            # Get user query patterns if available
            query_patterns = {}
            if user_id:
                query_patterns = await self.memory_manager.get_query_patterns(user_id)

            # Build context string
            context_str = self._build_context_string(
                refined_query,
                context
            )

            # Format query patterns
            patterns_str = self._format_query_patterns(query_patterns)

            # Generate routing decision
            prompt = ChatPromptTemplate.from_template(ROUTING_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            result = await chain.ainvoke({
                "query": refined_query or query,
                "context": context_str,
                "query_patterns": patterns_str
            })

            # Parse JSON
            try:
                if "```json" in result:
                    result = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    result = result.split("```")[1].split("```")[0].strip()

                routing_decision = json.loads(result)

            except json.JSONDecodeError:
                logger.warning("Failed to parse routing JSON, using default")
                routing_decision = {
                    "selected_agent": "lean_hybrid",
                    "confidence": 0.5,
                    "reasoning": "fallback to default agent",
                    "query_analysis": {
                        "complexity": "unknown",
                        "domain": "general",
                        "intent": "factual",
                        "citation_critical": False
                    }
                }

            # Validate agent availability
            selected_agent = routing_decision.get("selected_agent", "lean_hybrid")
            if selected_agent not in self.available_agents:
                logger.warning(
                    f"Selected agent '{selected_agent}' not available, using lean_hybrid"
                )
                routing_decision["selected_agent"] = "lean_hybrid"
                routing_decision["fallback"] = True

            logger.info(
                "Routing decision made",
                selected_agent=routing_decision["selected_agent"],
                confidence=routing_decision.get("confidence", 0),
                reasoning=routing_decision.get("reasoning", "")[:100]
            )

            return routing_decision

        except Exception as e:
            logger.error("Routing failed", error=str(e))
            # Fallback to default agent
            return {
                "selected_agent": "lean_hybrid",
                "confidence": 0.5,
                "reasoning": f"error fallback: {str(e)}",
                "query_analysis": {
                    "complexity": "unknown",
                    "domain": "general",
                    "intent": "factual",
                    "citation_critical": False
                },
                "error": True
            }

    def _build_context_string(
        self,
        refined_query: str | None,
        context: Dict[str, Any] | None
    ) -> str:
        """Build context string for routing."""
        parts = []

        if refined_query:
            parts.append(f"Refined Query: {refined_query}")

        if context:
            if context.get("facets"):
                parts.append(f"Topical Facets: {', '.join(context['facets'])}")
            if context.get("search_intents"):
                parts.append(f"Search Intents: {', '.join(context['search_intents'])}")

        if not parts:
            parts.append("No additional context")

        return "\n".join(parts)

    def _format_query_patterns(self, query_patterns: Dict[str, Any]) -> str:
        """Format query patterns for prompt."""
        if not query_patterns:
            return "No historical patterns available"

        formatted = []
        for agent, stats in query_patterns.items():
            if agent in self.available_agents:
                success_rate = (
                    stats["success_count"] / stats["count"]
                    if stats["count"] > 0
                    else 0
                )
                formatted.append(
                    f"- {agent}: {stats['count']} uses, {success_rate:.1%} success rate"
                )

        return "\n".join(formatted) if formatted else "No patterns yet"


def create_router() -> Router:
    """
    Create a Router instance.

    Returns:
        Router
    """
    return Router()
