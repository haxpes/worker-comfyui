"""
PlanThenReadAgent: Decomposes complex queries into sub-questions.
Best for multi-part questions requiring structured reasoning.
"""
import time
import json
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
from app.utils.prompts import format_context_with_citations
from app.utils.logger import get_logger

logger = get_logger(__name__)


DECOMPOSITION_PROMPT = """You are an expert at breaking down complex questions into simpler sub-questions.

Analyze the following question and decompose it into 2-5 independent sub-questions that, when answered, will provide a complete answer to the original question.

Original Question: {query}

Requirements:
1. Each sub-question should be clear and specific
2. Sub-questions should be independent (can be answered separately)
3. Together, they should cover all aspects of the original question
4. Keep sub-questions simple and focused

Output Format (JSON):
{{
  "sub_questions": [
    {{"question": "sub-question 1", "reasoning": "why this is needed"}},
    {{"question": "sub-question 2", "reasoning": "why this is needed"}}
  ]
}}

Decompose the question now:"""

SYNTHESIS_PROMPT = """You are an expert at synthesizing information from multiple sources.

Original Question: {original_query}

Sub-Questions and Answers:
{sub_answers}

Task: Create a comprehensive answer to the original question by synthesizing the sub-answers.

Requirements:
1. Integrate all sub-answers into a coherent response
2. Use [N] notation for citations from the context
3. Ensure logical flow and completeness
4. Address all aspects of the original question

Comprehensive Answer:"""


class PlanThenReadAgent(BaseAgent):
    """
    PlanThenReadAgent decomposes queries into sub-questions.

    Workflow:
    1. Decompose: Break query into sub-questions
    2. Retrieve: For each sub-question, retrieve relevant docs
    3. Answer: Answer each sub-question
    4. Synthesize: Combine sub-answers into final answer
    """

    def __init__(self):
        """Initialize PlanThenReadAgent."""
        super().__init__(name="plan_then_read")

        # Initialize retriever
        self.hybrid_retriever = create_hybrid_retriever()

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.complex_model,  # Use GPT-4 for better planning
            temperature=0,
            openai_api_key=settings.openai_api_key
        )

        logger.info("PlanThenReadAgent initialized")

    def build_graph(self) -> CompiledGraph:
        """
        Build the LangGraph workflow.

        Nodes:
        - decompose: Break query into sub-questions
        - retrieve_all: Retrieve docs for each sub-question
        - answer_subquestions: Answer each sub-question
        - synthesize: Combine into final answer

        Returns:
            Compiled LangGraph workflow
        """
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("decompose", self._decompose_node)
        workflow.add_node("retrieve_all", self._retrieve_all_node)
        workflow.add_node("answer_subquestions", self._answer_subquestions_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Define edges
        workflow.set_entry_point("decompose")
        workflow.add_edge("decompose", "retrieve_all")
        workflow.add_edge("retrieve_all", "answer_subquestions")
        workflow.add_edge("answer_subquestions", "synthesize")
        workflow.add_edge("synthesize", END)

        logger.info("PlanThenReadAgent graph built")

        return workflow.compile()

    async def _decompose_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Decompose query into sub-questions.

        Args:
            state: Current agent state

        Returns:
            Updated state with sub_questions
        """
        query = state["query"]
        start_time = time.time()

        logger.info("Decompose node started", query_preview=query[:100])

        try:
            # Create decomposition prompt
            prompt = ChatPromptTemplate.from_template(DECOMPOSITION_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            # Get decomposition
            result = await chain.ainvoke({"query": query})

            # Parse JSON
            try:
                # Extract JSON from result (might have markdown)
                if "```json" in result:
                    result = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    result = result.split("```")[1].split("```")[0].strip()

                decomposition = json.loads(result)
                sub_questions = decomposition.get("sub_questions", [])

            except json.JSONDecodeError:
                logger.warning("Failed to parse decomposition JSON, falling back to single question")
                sub_questions = [{"question": query, "reasoning": "original query"}]

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                "Decompose node completed",
                sub_questions=len(sub_questions),
                duration_ms=round(duration_ms, 2)
            )

            return {
                "metadata": {
                    "sub_questions": sub_questions
                }
            }

        except Exception as e:
            logger.error("Decompose node failed", error=str(e))
            # Fallback: treat as single question
            return {
                "metadata": {
                    "sub_questions": [{"question": query, "reasoning": "fallback"}]
                }
            }

    async def _retrieve_all_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieve documents for all sub-questions.

        Args:
            state: Current agent state

        Returns:
            Updated state with retrieved docs per sub-question
        """
        sub_questions = state.get("metadata", {}).get("sub_questions", [])
        start_time = time.time()

        logger.info("Retrieve all node started", num_sub_questions=len(sub_questions))

        try:
            # Retrieve for each sub-question
            all_retrievals = []

            for i, sq in enumerate(sub_questions):
                question = sq["question"]
                docs = await self.hybrid_retriever.retrieve(question)

                # Rerank
                reranked = await rerank_documents(
                    query=question,
                    documents=docs,
                    top_k=settings.mmr_k
                )

                all_retrievals.append({
                    "question": question,
                    "docs": reranked
                })

                logger.debug(f"Retrieved {len(reranked)} docs for sub-question {i+1}")

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                "Retrieve all node completed",
                duration_ms=round(duration_ms, 2)
            )

            # Store in metadata
            metadata = state.get("metadata", {})
            metadata["retrievals"] = all_retrievals

            return {
                "metadata": metadata
            }

        except Exception as e:
            logger.error("Retrieve all node failed", error=str(e))
            raise

    async def _answer_subquestions_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Answer each sub-question.

        Args:
            state: Current agent state

        Returns:
            Updated state with sub-answers
        """
        retrievals = state.get("metadata", {}).get("retrievals", [])
        start_time = time.time()

        logger.info("Answer subquestions node started", num_questions=len(retrievals))

        try:
            sub_answers = []

            for retrieval in retrievals:
                question = retrieval["question"]
                docs = retrieval["docs"]

                # Format context
                context = format_context_with_citations(docs)

                # Simple QA prompt
                prompt = ChatPromptTemplate.from_template(
                    "Answer the following question based on the context.\n\n"
                    "Question: {question}\n\nContext:\n{context}\n\nAnswer (with [N] citations):"
                )

                chain = prompt | self.llm | StrOutputParser()
                answer = await chain.ainvoke({
                    "question": question,
                    "context": context
                })

                sub_answers.append({
                    "question": question,
                    "answer": answer,
                    "docs": docs
                })

                logger.debug(f"Answered: {question[:50]}...")

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                "Answer subquestions node completed",
                duration_ms=round(duration_ms, 2)
            )

            # Store in metadata
            metadata = state.get("metadata", {})
            metadata["sub_answers"] = sub_answers

            return {
                "metadata": metadata
            }

        except Exception as e:
            logger.error("Answer subquestions node failed", error=str(e))
            raise

    async def _synthesize_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Synthesize final answer from sub-answers.

        Args:
            state: Current agent state

        Returns:
            Updated state with final answer
        """
        original_query = state["query"]
        sub_answers = state.get("metadata", {}).get("sub_answers", [])
        start_time = time.time()

        logger.info("Synthesize node started")

        try:
            # Format sub-answers
            sub_answers_text = "\n\n".join([
                f"Q: {sa['question']}\nA: {sa['answer']}"
                for sa in sub_answers
            ])

            # Synthesize
            prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            final_answer = await chain.ainvoke({
                "original_query": original_query,
                "sub_answers": sub_answers_text
            })

            # Collect all docs and citations
            all_docs = []
            for sa in sub_answers:
                all_docs.extend(sa["docs"])

            # Deduplicate docs
            seen_ids = set()
            unique_docs = []
            for doc in all_docs:
                doc_id = doc.metadata.get("id")
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_docs.append(doc)

            # Extract citations from final answer
            from app.utils.prompts import extract_citations_from_answer
            citations = extract_citations_from_answer(final_answer, unique_docs)

            # Confidence based on sub-answer quality
            confidence = 0.75  # Moderate confidence for multi-part

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                "Synthesize node completed",
                answer_length=len(final_answer),
                citations=len(citations),
                duration_ms=round(duration_ms, 2)
            )

            return {
                "answer": final_answer,
                "citations": citations,
                "confidence": confidence,
                "reranked_docs": unique_docs
            }

        except Exception as e:
            logger.error("Synthesize node failed", error=str(e))
            raise


def create_plan_then_read_agent() -> PlanThenReadAgent:
    """
    Create a PlanThenReadAgent instance.

    Returns:
        PlanThenReadAgent
    """
    return PlanThenReadAgent()
