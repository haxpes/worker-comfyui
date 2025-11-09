"""
EvidenceFirstAgent: Extractive QA with strict citation mapping.
Best for compliance, regulatory, and citation-critical tasks.
"""
import time
from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from app.config import settings
from app.core.state import AgentState
from app.agents.base import BaseAgent
from app.retrieval.hybrid import create_hybrid_retriever
from app.retrieval.reranker import rerank_documents
from app.utils.logger import get_logger

logger = get_logger(__name__)

EXTRACTION_PROMPT = """You are an expert at extracting exact text spans from documents.

Task: Extract EXACT text spans from the context that answer the question. Do not paraphrase or rephrase.

Question: {query}

Context:
{context}

Instructions:
1. Extract exact text spans (word-for-word from context)
2. Each span must directly answer some aspect of the question
3. Include the source document number [N] for each span
4. Rank spans by relevance

Output Format (JSON):
{{
  "spans": [
    {{
      "text": "exact text from context",
      "source": 1,
      "relevance": 0.0-1.0,
      "reasoning": "why this span is relevant"
    }}
  ]
}}

Extract spans now:"""

SYNTHESIS_PROMPT = """You are an expert at synthesizing answers from extracted evidence.

Question: {query}

Extracted Evidence Spans:
{spans}

Task: Create a clear, well-structured answer using ONLY the provided evidence spans.

Requirements:
1. Use [N] citations to reference source documents
2. Do NOT add information not present in the spans
3. Organize the answer logically
4. Ensure every claim is directly supported by a span
5. If evidence is incomplete, state what's missing

Answer:"""


class EvidenceFirstAgent(BaseAgent):
    """
    EvidenceFirstAgent with extractive QA and strict citation mapping.

    Workflow:
    1. Retrieve: Hybrid retrieval + reranking
    2. Extract: Extract exact evidence spans
    3. Synthesize: Generate answer from spans only
    """

    def __init__(self):
        """Initialize EvidenceFirstAgent."""
        super().__init__(name="evidence_first")

        # Initialize retriever
        self.hybrid_retriever = create_hybrid_retriever()

        # Initialize LLM (use GPT-4 for better extraction)
        self.llm = ChatOpenAI(
            model=settings.complex_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )

        logger.info("EvidenceFirstAgent initialized")

    def build_graph(self) -> CompiledGraph:
        """
        Build the LangGraph workflow.

        Nodes:
        - retrieve: Hybrid retrieval + reranking
        - extract: Extract exact evidence spans
        - synthesize: Generate answer from spans

        Returns:
            Compiled LangGraph workflow
        """
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("extract", self._extract_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Define edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "extract")
        workflow.add_edge("extract", "synthesize")
        workflow.add_edge("synthesize", END)

        logger.info("EvidenceFirstAgent graph built")

        return workflow.compile()

    async def _retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieve and rerank documents.

        Args:
            state: Current agent state

        Returns:
            Updated state with retrieved docs
        """
        query = state["query"]
        start_time = time.time()

        logger.info("Retrieve node started", query_preview=query[:100])

        try:
            # Hybrid retrieval
            docs = await self.hybrid_retriever.retrieve(query)

            # Rerank with higher k for extraction
            reranked_docs = await rerank_documents(
                query=query,
                documents=docs,
                top_k=min(10, len(docs))  # More docs for extraction
            )

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                "Retrieve node completed",
                results=len(reranked_docs),
                duration_ms=round(duration_ms, 2)
            )

            return {
                "reranked_docs": reranked_docs,
            }

        except Exception as e:
            logger.error("Retrieve node failed", error=str(e))
            raise

    async def _extract_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Extract exact evidence spans from documents.

        Args:
            state: Current agent state

        Returns:
            Updated state with extracted spans
        """
        query = state["query"]
        docs = state["reranked_docs"]
        start_time = time.time()

        logger.info("Extract node started", num_docs=len(docs))

        try:
            # Format context
            from app.utils.prompts import format_context_with_citations
            context = format_context_with_citations(docs)

            # Extract spans
            prompt = ChatPromptTemplate.from_template(EXTRACTION_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            result = await chain.ainvoke({
                "query": query,
                "context": context
            })

            # Parse JSON
            import json
            try:
                if "```json" in result:
                    result = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    result = result.split("```")[1].split("```")[0].strip()

                extraction = json.loads(result)
                spans = extraction.get("spans", [])

            except json.JSONDecodeError:
                logger.warning("Failed to parse extraction JSON")
                spans = []

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                "Extract node completed",
                spans=len(spans),
                duration_ms=round(duration_ms, 2)
            )

            # Store in metadata
            return {
                "metadata": {
                    "extracted_spans": spans
                }
            }

        except Exception as e:
            logger.error("Extract node failed", error=str(e))
            raise

    async def _synthesize_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Synthesize answer from extracted spans.

        Args:
            state: Current agent state

        Returns:
            Updated state with answer
        """
        query = state["query"]
        spans = state.get("metadata", {}).get("extracted_spans", [])
        docs = state["reranked_docs"]
        start_time = time.time()

        logger.info("Synthesize node started", spans=len(spans))

        try:
            if not spans:
                # Fallback: no extraction, use simple answer
                answer = "Insufficient evidence found to answer the question with high confidence."
                citations = []
                confidence = 0.3
            else:
                # Format spans
                spans_text = "\n\n".join([
                    f"[{s['source']}] \"{s['text']}\"\n(Relevance: {s['relevance']}, Reason: {s['reasoning']})"
                    for s in spans
                ])

                # Synthesize from spans
                prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)
                chain = prompt | self.llm | StrOutputParser()

                answer = await chain.ainvoke({
                    "query": query,
                    "spans": spans_text
                })

                # Extract citations
                from app.utils.prompts import extract_citations_from_answer
                citations = extract_citations_from_answer(answer, docs)

                # High confidence for evidence-based answers
                confidence = min(0.95, 0.7 + (len(citations) * 0.05))

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                "Synthesize node completed",
                answer_length=len(answer),
                citations=len(citations) if isinstance(citations, list) else 0,
                confidence=confidence,
                duration_ms=round(duration_ms, 2)
            )

            return {
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
            }

        except Exception as e:
            logger.error("Synthesize node failed", error=str(e))
            raise


def create_evidence_first_agent() -> EvidenceFirstAgent:
    """
    Create an EvidenceFirstAgent instance.

    Returns:
        EvidenceFirstAgent
    """
    return EvidenceFirstAgent()
