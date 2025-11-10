"""
DistillFirstAgent: Fast agent using pre-distilled knowledge with corpus fallback.
Checks conversation summaries and similar queries before full retrieval.
"""
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from app.config import settings
from app.core.state import AgentState, Citation
from app.agents.base import BaseAgent
from app.core.memory import get_memory_manager
from app.retrieval.hybrid import create_hybrid_retriever
from app.retrieval.reranker import get_reranker
from app.utils.logger import get_logger

logger = get_logger(__name__)

DISTILLED_SYNTHESIS_PROMPT = """You are an AI assistant leveraging pre-distilled knowledge for fast responses.

Question: {query}

Pre-Distilled Knowledge:
{distilled_knowledge}

Confidence in Distilled Knowledge: {distilled_confidence}

Task: Answer the question using the distilled knowledge.
If confidence is low or knowledge incomplete, indicate what additional information is needed.

Answer:"""

CONFLICT_RESOLUTION_PROMPT = """You are resolving conflicts between pre-distilled knowledge and fresh corpus retrieval.

Question: {query}

Pre-Distilled Knowledge:
{distilled_knowledge}

Fresh Corpus Evidence:
{corpus_evidence}

Task: Synthesize a final answer that:
1. Prefers fresh evidence for factual accuracy
2. Uses distilled knowledge for context and background
3. Flags any conflicts and explains resolution
4. Provides updated understanding

Answer:"""


class DistillFirstAgent(BaseAgent):
    """
    Agent that checks distilled knowledge first, falls back to corpus if needed.
    Fast for repeated or similar queries.
    """

    def __init__(
        self,
        corpus_id: Optional[str] = None,
        distilled_threshold: float = 0.75,
        similarity_k: int = 3,
        retrieval_k: int = 10
    ):
        """
        Initialize DistillFirstAgent.

        Args:
            corpus_id: Optional corpus ID
            distilled_threshold: Confidence threshold to use only distilled knowledge
            similarity_k: Number of similar conversations to check
            retrieval_k: Documents to retrieve if fallback needed
        """
        super().__init__(name="distill_first")
        self.corpus_id = corpus_id
        self.distilled_threshold = distilled_threshold
        self.similarity_k = similarity_k
        self.retrieval_k = retrieval_k

        # Memory for distilled knowledge
        self.memory = get_memory_manager()

        # Retriever for fallback
        self.retriever = create_hybrid_retriever(
            corpus_id=corpus_id,
            bm25_k=retrieval_k,
            dense_k=retrieval_k
        )

        self.reranker = get_reranker()

        # LLM
        self.llm = ChatOpenAI(
            model=settings.default_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )

        self._graph = self.build_graph()

        logger.info(
            "DistillFirstAgent initialized",
            corpus_id=corpus_id,
            threshold=distilled_threshold
        )

    def build_graph(self) -> CompiledGraph:
        """
        Build the agent's LangGraph workflow.

        Returns:
            Compiled graph
        """
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("check_distilled", self._check_distilled_node)
        workflow.add_node("retrieve_corpus", self._retrieve_corpus_node)
        workflow.add_node("resolve_conflicts", self._resolve_conflicts_node)
        workflow.add_node("synthesize_distilled", self._synthesize_distilled_node)

        # Define edges
        workflow.set_entry_point("check_distilled")

        # Conditional: use distilled or retrieve from corpus
        workflow.add_conditional_edges(
            "check_distilled",
            self._should_use_distilled,
            {
                "distilled": "synthesize_distilled",
                "corpus": "retrieve_corpus"
            }
        )

        workflow.add_edge("synthesize_distilled", END)
        workflow.add_edge("retrieve_corpus", "resolve_conflicts")
        workflow.add_edge("resolve_conflicts", END)

        return workflow.compile()

    async def _check_distilled_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Check for pre-distilled knowledge from similar queries.

        Args:
            state: Current state

        Returns:
            State updates
        """
        query = state["query"]

        logger.info("Checking distilled knowledge", query_preview=query[:100])

        try:
            # Find similar conversation summaries
            similar_summaries = await self.memory.find_similar_conversations(
                query=query,
                k=self.similarity_k
            )

            if similar_summaries:
                # Build distilled knowledge from summaries
                distilled_parts = []
                for i, (summary, similarity) in enumerate(similar_summaries, 1):
                    distilled_parts.append(
                        f"[Similar Query {i}] (Similarity: {similarity:.2f})\n{summary}"
                    )

                distilled_knowledge = "\n\n".join(distilled_parts)

                # Estimate confidence based on similarity scores
                avg_similarity = sum(s for _, s in similar_summaries) / len(similar_summaries)
                distilled_confidence = avg_similarity

                logger.info(
                    f"Found {len(similar_summaries)} similar conversations",
                    avg_similarity=avg_similarity
                )

                return {
                    "distilled_knowledge": distilled_knowledge,
                    "distilled_confidence": distilled_confidence,
                    "has_distilled": True
                }

            else:
                logger.info("No similar conversations found")
                return {
                    "distilled_knowledge": "",
                    "distilled_confidence": 0.0,
                    "has_distilled": False
                }

        except Exception as e:
            logger.error("Distilled knowledge check failed", error=str(e))
            return {
                "distilled_knowledge": "",
                "distilled_confidence": 0.0,
                "has_distilled": False
            }

    def _should_use_distilled(self, state: AgentState) -> str:
        """
        Decide whether to use distilled knowledge or retrieve from corpus.

        Args:
            state: Current state

        Returns:
            Next node name
        """
        has_distilled = state.get("has_distilled", False)
        confidence = state.get("distilled_confidence", 0.0)

        if has_distilled and confidence >= self.distilled_threshold:
            logger.info(
                f"Using distilled knowledge (confidence={confidence:.2f})"
            )
            return "distilled"
        else:
            logger.info(
                f"Falling back to corpus retrieval (confidence={confidence:.2f})"
            )
            return "corpus"

    async def _synthesize_distilled_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Synthesize answer from distilled knowledge only.

        Args:
            state: Current state

        Returns:
            State updates with answer
        """
        query = state["query"]
        distilled_knowledge = state.get("distilled_knowledge", "")
        confidence = state.get("distilled_confidence", 0.0)

        logger.info("Synthesizing from distilled knowledge")

        # Generate answer
        prompt = ChatPromptTemplate.from_template(DISTILLED_SYNTHESIS_PROMPT)
        chain = prompt | self.llm | StrOutputParser()

        answer = await chain.ainvoke({
            "query": query,
            "distilled_knowledge": distilled_knowledge,
            "distilled_confidence": f"{confidence:.2f}"
        })

        # Citations are from similar conversations (metadata)
        citations = [
            Citation(
                citation_number=i,
                content="Pre-distilled knowledge from similar query",
                doc_id="distilled",
                metadata={"source": "conversation_summary", "type": "distilled"}
            )
            for i in range(1, min(self.similarity_k + 1, 4))
        ]

        logger.info(
            "Distilled synthesis completed",
            confidence=confidence
        )

        return {
            "answer": answer,
            "citations": [c.dict() for c in citations],
            "confidence": confidence,
            "metadata": {
                "source": "distilled_knowledge",
                "fallback_used": False
            }
        }

    async def _retrieve_corpus_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieve from corpus (fallback).

        Args:
            state: Current state

        Returns:
            State updates
        """
        query = state["query"]

        logger.info("Retrieving from corpus", query_preview=query[:100])

        try:
            # Retrieve
            docs = await self.retriever.ainvoke(query)

            # Rerank
            if docs:
                reranked_docs = await self.reranker.arerank(
                    query=query,
                    documents=docs,
                    top_k=self.retrieval_k // 2
                )
            else:
                reranked_docs = []

            logger.info(f"Retrieved {len(reranked_docs)} corpus documents")

            return {
                "corpus_docs": reranked_docs
            }

        except Exception as e:
            logger.error("Corpus retrieval failed", error=str(e))
            return {"corpus_docs": []}

    async def _resolve_conflicts_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Resolve conflicts between distilled knowledge and corpus evidence.

        Args:
            state: Current state

        Returns:
            State updates with answer
        """
        query = state["query"]
        distilled_knowledge = state.get("distilled_knowledge", "")
        corpus_docs = state.get("corpus_docs", [])

        if not corpus_docs:
            # No corpus evidence, must use distilled if available
            if distilled_knowledge:
                return await self._synthesize_distilled_node(state)
            else:
                return {
                    "answer": "I couldn't find relevant information to answer your question.",
                    "citations": [],
                    "confidence": 0.0
                }

        # Build corpus evidence
        corpus_evidence = "\n\n".join([
            f"[{i+1}] {doc.page_content}"
            for i, doc in enumerate(corpus_docs)
        ])

        # Resolve conflicts
        prompt = ChatPromptTemplate.from_template(CONFLICT_RESOLUTION_PROMPT)
        chain = prompt | self.llm | StrOutputParser()

        answer = await chain.ainvoke({
            "query": query,
            "distilled_knowledge": distilled_knowledge or "No pre-distilled knowledge available",
            "corpus_evidence": corpus_evidence
        })

        # Build citations from corpus
        citations = [
            Citation(
                citation_number=i,
                content=doc.page_content[:200],
                doc_id=doc.metadata.get("doc_id", "unknown"),
                metadata=doc.metadata
            )
            for i, doc in enumerate(corpus_docs, 1)
        ]

        logger.info(
            "Conflict resolution completed",
            distilled_available=bool(distilled_knowledge),
            corpus_docs=len(corpus_docs)
        )

        return {
            "answer": answer,
            "citations": [c.dict() for c in citations],
            "confidence": 0.85,  # High confidence when using fresh corpus evidence
            "sources": corpus_docs,
            "metadata": {
                "source": "corpus_with_distilled_context",
                "fallback_used": True
            }
        }


def create_distill_first_agent(corpus_id: Optional[str] = None, **kwargs) -> DistillFirstAgent:
    """
    Create a DistillFirstAgent instance.

    Args:
        corpus_id: Optional corpus ID
        **kwargs: Additional arguments

    Returns:
        DistillFirstAgent
    """
    return DistillFirstAgent(corpus_id=corpus_id, **kwargs)
