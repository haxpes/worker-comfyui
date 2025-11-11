"""
QDMR Decomposition and Evidence Hypothesis Generation.

Decomposes queries into atomic sub-questions (QDMR-style) and converts
them into declarative evidence hypotheses for precise NLI checks.
"""
import json
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from app.config import settings
from app.qfsd.models import QDMRPlan, SubQuestion, EvidenceHypothesis
from app.utils.logger import get_logger

logger = get_logger(__name__)


QDMR_DECOMPOSITION_PROMPT = """You are an expert at breaking down complex questions using QDMR (Question Decomposition Meaning Representation).

Analyze the following question and decompose it into 2-5 atomic sub-questions that:
1. Are independent (can be answered separately)
2. Cover all aspects of the original question
3. Form a logical dependency graph

Original Question: {query}

Output Format (JSON):
{{
  "sub_questions": [
    {{
      "subq_id": "subq_1",
      "question": "What is X?",
      "reasoning": "Need to define X first",
      "info_needs": ["X definition"],
      "dependencies": []
    }},
    {{
      "subq_id": "subq_2",
      "question": "What are the requirements for X?",
      "reasoning": "Need requirements after understanding X",
      "info_needs": ["X requirements"],
      "dependencies": ["subq_1"]
    }}
  ]
}}

Decompose the question now:"""


HYPOTHESIS_GENERATION_PROMPT = """Convert the following sub-question into a declarative evidence hypothesis for NLI checking.

Sub-question: {sub_question}
Information Need: {info_need}

Rules:
1. Convert question to declarative statement
2. Make it testable via entailment
3. Be specific and atomic
4. Keep it factual, not subjective

Example:
Question: "What are MDDS FDA requirements?"
Hypothesis: "MDDS requires FDA premarket submission"

Generate the declarative hypothesis:"""


class QDMRDecomposer:
    """
    QDMR decomposition with evidence hypothesis generation.
    """

    def __init__(self, model_name: str | None = None):
        """
        Initialize QDMR decomposer.

        Args:
            model_name: LLM model name (defaults to config)
        """
        self.llm = ChatOpenAI(
            model=model_name or settings.complex_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )
        logger.info("QDMRDecomposer initialized")

    async def decompose(self, query: str) -> QDMRPlan:
        """
        Decompose query into QDMR-style atomic sub-questions.

        Args:
            query: Query to decompose

        Returns:
            QDMRPlan with sub-questions and evidence hypotheses
        """
        logger.info("Decomposing query", query_preview=query[:100])

        try:
            # Step 1: QDMR decomposition
            prompt = ChatPromptTemplate.from_template(QDMR_DECOMPOSITION_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            result = await chain.ainvoke({"query": query})

            # Parse JSON
            try:
                # Extract JSON from result (might have markdown)
                if "```json" in result:
                    result = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    result = result.split("```")[1].split("```")[0].strip()

                decomposition = json.loads(result)
                sub_questions_data = decomposition.get("sub_questions", [])

            except json.JSONDecodeError as e:
                logger.warning(
                    "Failed to parse QDMR JSON, falling back to single question",
                    error=str(e)
                )
                sub_questions_data = [{
                    "subq_id": "subq_1",
                    "question": query,
                    "reasoning": "fallback",
                    "info_needs": [query],
                    "dependencies": []
                }]

            # Convert to SubQuestion objects
            sub_questions = [
                SubQuestion(**sq) for sq in sub_questions_data
            ]

            # Step 2: Generate evidence hypotheses
            evidence_hypotheses = await self._generate_hypotheses(
                query=query,
                sub_questions=sub_questions
            )

            # Build dependency graph
            dependency_graph = {
                sq.subq_id: sq.dependencies
                for sq in sub_questions
            }

            logger.info(
                "QDMR decomposition completed",
                sub_questions=len(sub_questions),
                hypotheses=len(evidence_hypotheses)
            )

            return QDMRPlan(
                query=query,
                sub_questions=sub_questions,
                evidence_hypotheses=evidence_hypotheses,
                dependency_graph=dependency_graph
            )

        except Exception as e:
            logger.error("QDMR decomposition failed", error=str(e))
            # Fallback: single question
            return QDMRPlan(
                query=query,
                sub_questions=[SubQuestion(
                    subq_id="subq_1",
                    question=query,
                    reasoning="fallback",
                    info_needs=[query],
                    dependencies=[]
                )],
                evidence_hypotheses=[EvidenceHypothesis(
                    hypothesis_id="hyp_1",
                    hypothesis=query,
                    information_need=query,
                    sub_question=query,
                    dependencies=[]
                )],
                dependency_graph={}
            )

    async def _generate_hypotheses(
        self,
        query: str,
        sub_questions: List[SubQuestion]
    ) -> List[EvidenceHypothesis]:
        """
        Generate declarative evidence hypotheses from sub-questions.

        Args:
            query: Original query
            sub_questions: Sub-questions from QDMR

        Returns:
            List of evidence hypotheses
        """
        hypotheses = []

        prompt = ChatPromptTemplate.from_template(HYPOTHESIS_GENERATION_PROMPT)
        chain = prompt | self.llm | StrOutputParser()

        for i, subq in enumerate(sub_questions):
            # Generate hypothesis for each information need
            for j, info_need in enumerate(subq.info_needs):
                try:
                    hypothesis_text = await chain.ainvoke({
                        "sub_question": subq.question,
                        "info_need": info_need
                    })

                    # Clean up response
                    hypothesis_text = hypothesis_text.strip()
                    if hypothesis_text.startswith('"') and hypothesis_text.endswith('"'):
                        hypothesis_text = hypothesis_text[1:-1]

                    # Create hypothesis
                    hyp = EvidenceHypothesis(
                        hypothesis_id=f"{subq.subq_id}_hyp_{j+1}",
                        hypothesis=hypothesis_text,
                        information_need=info_need,
                        sub_question=subq.question,
                        dependencies=subq.dependencies,
                        is_explicit=True  # All QDMR-generated are explicit
                    )

                    hypotheses.append(hyp)

                    logger.debug(
                        "Generated hypothesis",
                        subq_id=subq.subq_id,
                        hypothesis=hypothesis_text[:50]
                    )

                except Exception as e:
                    logger.warning(
                        "Failed to generate hypothesis, using fallback",
                        error=str(e),
                        subq_id=subq.subq_id
                    )
                    # Fallback: use sub-question as hypothesis
                    hypotheses.append(EvidenceHypothesis(
                        hypothesis_id=f"{subq.subq_id}_hyp_{j+1}",
                        hypothesis=subq.question,
                        information_need=info_need,
                        sub_question=subq.question,
                        dependencies=subq.dependencies
                    ))

        return hypotheses


def create_qdmr_decomposer(model_name: str | None = None) -> QDMRDecomposer:
    """
    Create a QDMR decomposer instance.

    Args:
        model_name: Optional model name

    Returns:
        QDMRDecomposer
    """
    return QDMRDecomposer(model_name=model_name)
