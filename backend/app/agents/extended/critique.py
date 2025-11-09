"""
CritiqueAgent: Post-synthesis quality assurance and improvement.
Reviews answers for completeness, alternative perspectives, and contradictions.
"""
import json
from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

CRITIQUE_PROMPT = """You are an expert reviewer analyzing the quality of an answer.

Original Question: {query}

Answer: {answer}

Source Documents:
{sources}

Task: Critique this answer for quality, completeness, and accuracy.

Evaluate:
1. **Completeness**: Does it address all aspects of the question?
2. **Alternative Perspectives**: Are there other valid viewpoints not mentioned?
3. **Contradictions**: Are there any internal contradictions or conflicts with sources?
4. **Missing Information**: What important information is missing?
5. **Clarity**: Is the answer clear and well-organized?

Output Format (JSON):
{{
  "completeness_score": 0.0-1.0,
  "clarity_score": 0.0-1.0,
  "issues": ["issue1", "issue2"],
  "missing_information": ["missing1", "missing2"],
  "alternative_perspectives": ["perspective1"],
  "suggested_improvements": ["improvement1", "improvement2"],
  "overall_quality": 0.0-1.0
}}

Provide your critique now:"""

IMPROVEMENT_PROMPT = """You are an expert at improving answers based on critique feedback.

Original Question: {query}

Original Answer: {answer}

Critique Feedback:
{critique}

Source Documents:
{sources}

Task: Improve the answer based on the critique feedback.

Requirements:
1. Address the issues identified
2. Add missing information if available in sources
3. Incorporate alternative perspectives
4. Maintain citation integrity (use [N] notation)
5. Improve clarity and organization

Improved Answer:"""


class CritiqueAgent:
    """
    CritiqueAgent for post-synthesis quality assurance.
    Reviews and optionally improves answers.
    """

    def __init__(self):
        """Initialize CritiqueAgent."""
        self.llm = ChatOpenAI(
            model=settings.complex_model,  # Use GPT-4 for better critique
            temperature=0,
            openai_api_key=settings.openai_api_key
        )

        logger.info("CritiqueAgent initialized")

    async def critique(
        self,
        query: str,
        answer: str,
        sources: List[Document]
    ) -> Dict[str, Any]:
        """
        Critique an answer.

        Args:
            query: Original question
            answer: Generated answer
            sources: Source documents

        Returns:
            Critique result dictionary
        """
        logger.info("Critiquing answer", query_preview=query[:50])

        try:
            # Format sources
            sources_text = "\n\n".join([
                f"[{i+1}] {doc.page_content[:500]}..."
                for i, doc in enumerate(sources[:5])  # Limit for token budget
            ])

            # Generate critique
            prompt = ChatPromptTemplate.from_template(CRITIQUE_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            result = await chain.ainvoke({
                "query": query,
                "answer": answer,
                "sources": sources_text
            })

            # Parse JSON
            try:
                if "```json" in result:
                    result = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    result = result.split("```")[1].split("```")[0].strip()

                critique = json.loads(result)

            except json.JSONDecodeError:
                logger.warning("Failed to parse critique JSON")
                critique = {
                    "completeness_score": 0.7,
                    "clarity_score": 0.7,
                    "issues": [],
                    "missing_information": [],
                    "alternative_perspectives": [],
                    "suggested_improvements": [],
                    "overall_quality": 0.7
                }

            logger.info(
                "Critique completed",
                overall_quality=critique.get("overall_quality", 0),
                issues=len(critique.get("issues", []))
            )

            return critique

        except Exception as e:
            logger.error("Critique failed", error=str(e))
            return {
                "completeness_score": 0.5,
                "clarity_score": 0.5,
                "issues": [str(e)],
                "missing_information": [],
                "alternative_perspectives": [],
                "suggested_improvements": [],
                "overall_quality": 0.5
            }

    async def improve(
        self,
        query: str,
        answer: str,
        critique: Dict[str, Any],
        sources: List[Document]
    ) -> str:
        """
        Improve an answer based on critique.

        Args:
            query: Original question
            answer: Original answer
            critique: Critique result
            sources: Source documents

        Returns:
            Improved answer
        """
        logger.info("Improving answer based on critique")

        try:
            # Format sources
            sources_text = "\n\n".join([
                f"[{i+1}] {doc.page_content[:500]}..."
                for i, doc in enumerate(sources[:5])
            ])

            # Format critique
            critique_text = json.dumps(critique, indent=2)

            # Generate improvement
            prompt = ChatPromptTemplate.from_template(IMPROVEMENT_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            improved_answer = await chain.ainvoke({
                "query": query,
                "answer": answer,
                "critique": critique_text,
                "sources": sources_text
            })

            logger.info(
                "Answer improved",
                original_length=len(answer),
                improved_length=len(improved_answer)
            )

            return improved_answer

        except Exception as e:
            logger.error("Answer improvement failed", error=str(e))
            return answer  # Return original on error

    async def critique_and_improve(
        self,
        query: str,
        answer: str,
        sources: List[Document],
        improvement_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Critique and conditionally improve an answer.

        Args:
            query: Original question
            answer: Generated answer
            sources: Source documents
            improvement_threshold: Quality threshold below which to improve

        Returns:
            Result with critique and potentially improved answer
        """
        # Generate critique
        critique = await self.critique(query, answer, sources)

        # Check if improvement needed
        overall_quality = critique.get("overall_quality", 1.0)

        if overall_quality < improvement_threshold:
            logger.info(
                "Quality below threshold, improving answer",
                quality=overall_quality,
                threshold=improvement_threshold
            )

            improved_answer = await self.improve(query, answer, critique, sources)

            return {
                "original_answer": answer,
                "improved_answer": improved_answer,
                "critique": critique,
                "was_improved": True
            }
        else:
            logger.info(
                "Quality acceptable, no improvement needed",
                quality=overall_quality
            )

            return {
                "original_answer": answer,
                "improved_answer": answer,
                "critique": critique,
                "was_improved": False
            }


def create_critique_agent() -> CritiqueAgent:
    """
    Create a CritiqueAgent instance.

    Returns:
        CritiqueAgent
    """
    return CritiqueAgent()
