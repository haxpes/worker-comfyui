"""
Neuro-Symbolic Validator: Verify answer quality and factuality.
Performs claim-level verification with re-retrieval for low confidence.
"""
import json
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from app.config import settings
from app.retrieval.hybrid import create_hybrid_retriever
from app.utils.logger import get_logger

logger = get_logger(__name__)

VALIDATION_PROMPT = """You are a rigorous fact-checker analyzing answer quality and factuality.

Question: {query}

Answer: {answer}

Source Documents:
{sources}

Task: Validate the answer across multiple dimensions. Output JSON:
{{
  "factuality": 0.0-1.0,  // Are claims supported by sources?
  "consistency": 0.0-1.0,  // Are claims internally consistent?
  "completeness": 0.0-1.0,  // Does answer fully address question?
  "citation_integrity": 0.0-1.0,  // Are citations accurate?
  "overall_score": 0.0-1.0,
  "issues": [
    {{
      "type": "factuality|consistency|completeness|citation",
      "severity": "low|medium|high",
      "description": "issue description",
      "claim": "problematic claim"
    }}
  ],
  "verified_claims": ["claim1", "claim2"],
  "unverified_claims": ["claim1", "claim2"]
}}

Validation:"""

CLAIM_EXTRACTION_PROMPT = """Extract distinct factual claims from this answer.

Answer: {answer}

Output JSON:
{{
  "claims": [
    {{
      "claim": "factual statement",
      "importance": "high|medium|low"
    }}
  ]
}}

Claims:"""


class Validator:
    """
    Validate answer quality through multi-dimensional analysis.
    Re-retrieves evidence for low-confidence claims.
    """

    def __init__(
        self,
        factuality_threshold: float = 0.7,
        consistency_threshold: float = 0.7,
        completeness_threshold: float = 0.6,
        enable_re_retrieval: bool = True
    ):
        """
        Initialize validator.

        Args:
            factuality_threshold: Minimum factuality score
            consistency_threshold: Minimum consistency score
            completeness_threshold: Minimum completeness score
            enable_re_retrieval: Whether to re-retrieve for low confidence claims
        """
        self.factuality_threshold = factuality_threshold
        self.consistency_threshold = consistency_threshold
        self.completeness_threshold = completeness_threshold
        self.enable_re_retrieval = enable_re_retrieval

        self.llm = ChatOpenAI(
            model=settings.default_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )

        logger.info(
            "Validator initialized",
            factuality_threshold=factuality_threshold,
            consistency_threshold=consistency_threshold
        )

    async def validate(
        self,
        query: str,
        answer: str,
        sources: List[Document],
        corpus_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate an answer.

        Args:
            query: Original query
            answer: Answer to validate
            sources: Source documents
            corpus_id: Optional corpus ID for re-retrieval

        Returns:
            Validation results dictionary
        """
        logger.info("Validating answer", query_preview=query[:100])

        # Format sources
        sources_text = "\n\n".join([
            f"[{i+1}] {doc.page_content[:300]}"
            for i, doc in enumerate(sources[:10])
        ])

        # Run validation
        try:
            prompt = ChatPromptTemplate.from_template(VALIDATION_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            validation_result = await chain.ainvoke({
                "query": query,
                "answer": answer,
                "sources": sources_text
            })

            # Parse JSON
            if "```json" in validation_result:
                validation_result = validation_result.split("```json")[1].split("```")[0].strip()
            elif "```" in validation_result:
                validation_result = validation_result.split("```")[1].split("```")[0].strip()

            validation = json.loads(validation_result)

            logger.info(
                "Validation completed",
                factuality=validation.get("factuality", 0),
                consistency=validation.get("consistency", 0),
                completeness=validation.get("completeness", 0),
                overall=validation.get("overall_score", 0)
            )

            # Check if re-retrieval needed
            if self.enable_re_retrieval and corpus_id:
                unverified_claims = validation.get("unverified_claims", [])

                if unverified_claims and validation.get("factuality", 1.0) < self.factuality_threshold:
                    logger.info(f"Re-retrieving evidence for {len(unverified_claims)} unverified claims")
                    additional_evidence = await self._re_retrieve_for_claims(
                        claims=unverified_claims,
                        corpus_id=corpus_id
                    )
                    validation["additional_evidence"] = additional_evidence

            return validation

        except Exception as e:
            logger.error("Validation failed", error=str(e))
            # Return default validation
            return {
                "factuality": 0.5,
                "consistency": 0.5,
                "completeness": 0.5,
                "citation_integrity": 0.5,
                "overall_score": 0.5,
                "issues": [{
                    "type": "validation_error",
                    "severity": "high",
                    "description": f"Validation error: {str(e)}",
                    "claim": "N/A"
                }],
                "verified_claims": [],
                "unverified_claims": []
            }

    async def extract_claims(self, answer: str) -> List[Dict[str, Any]]:
        """
        Extract factual claims from an answer.

        Args:
            answer: Answer text

        Returns:
            List of claims
        """
        try:
            prompt = ChatPromptTemplate.from_template(CLAIM_EXTRACTION_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            result = await chain.ainvoke({"answer": answer})

            # Parse JSON
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            claims_data = json.loads(result)

            return claims_data.get("claims", [])

        except Exception as e:
            logger.warning("Claim extraction failed", error=str(e))
            return []

    async def _re_retrieve_for_claims(
        self,
        claims: List[str],
        corpus_id: str,
        k_per_claim: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Re-retrieve evidence for unverified claims.

        Args:
            claims: List of claims to verify
            corpus_id: Corpus ID
            k_per_claim: Documents to retrieve per claim

        Returns:
            Additional evidence
        """
        retriever = create_hybrid_retriever(
            corpus_id=corpus_id,
            bm25_k=k_per_claim,
            dense_k=k_per_claim
        )

        additional_evidence = []

        for claim in claims[:5]:  # Limit to 5 claims
            try:
                docs = await retriever.ainvoke(claim)

                additional_evidence.append({
                    "claim": claim,
                    "evidence": [
                        {
                            "content": doc.page_content[:200],
                            "metadata": doc.metadata
                        }
                        for doc in docs[:k_per_claim]
                    ]
                })

            except Exception as e:
                logger.warning(f"Re-retrieval failed for claim", claim=claim[:50], error=str(e))

        return additional_evidence

    def is_valid(self, validation: Dict[str, Any]) -> bool:
        """
        Check if validation passes all thresholds.

        Args:
            validation: Validation results

        Returns:
            True if valid
        """
        factuality = validation.get("factuality", 0.0)
        consistency = validation.get("consistency", 0.0)
        completeness = validation.get("completeness", 0.0)

        return (
            factuality >= self.factuality_threshold and
            consistency >= self.consistency_threshold and
            completeness >= self.completeness_threshold
        )

    def get_quality_summary(self, validation: Dict[str, Any]) -> str:
        """
        Generate human-readable quality summary.

        Args:
            validation: Validation results

        Returns:
            Summary string
        """
        factuality = validation.get("factuality", 0.0)
        consistency = validation.get("consistency", 0.0)
        completeness = validation.get("completeness", 0.0)
        overall = validation.get("overall_score", 0.0)

        issues = validation.get("issues", [])
        high_issues = [i for i in issues if i.get("severity") == "high"]

        summary_parts = [
            f"Overall Quality: {overall:.2f}",
            f"Factuality: {factuality:.2f}",
            f"Consistency: {consistency:.2f}",
            f"Completeness: {completeness:.2f}"
        ]

        if high_issues:
            summary_parts.append(f"High-severity issues: {len(high_issues)}")

        return " | ".join(summary_parts)


def create_validator(**kwargs) -> Validator:
    """
    Create a Validator instance.

    Args:
        **kwargs: Configuration parameters

    Returns:
        Validator
    """
    return Validator(**kwargs)
