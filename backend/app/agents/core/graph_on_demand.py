"""
GraphOnDemandAgent: Relationship-aware retrieval using knowledge graph.
Constructs ephemeral subgraphs for query context.
"""
from typing import Dict, Any, Optional, List
from uuid import uuid4
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from app.config import settings
from app.core.state import AgentState, AgentResult, Citation
from app.agents.base import BaseAgent
from app.retrieval.entity import create_entity_retriever
from app.retrieval.community import create_community_retriever
from app.retrieval.reranker import get_reranker
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYNTHESIS_PROMPT = """You are an AI assistant specializing in relationship-aware question answering.

Question: {query}

Retrieved Context (entity-focused and community-based):
{context}

Entity Relationships:
{relationships}

Task: Answer the question using the provided context and entity relationships.
Focus on:
1. Direct answers from the context
2. Relationships between entities mentioned
3. Community/topic connections
4. Comprehensive coverage using graph structure

Provide a clear, well-structured answer with proper citations.

Answer:"""


class GraphOnDemandAgent(BaseAgent):
    """
    Agent using entity and community retrievers for relationship-aware QA.
    """

    def __init__(
        self,
        corpus_id: Optional[str] = None,
        entity_k: int = 10,
        community_k: int = 3,
        rerank_k: int = 5
    ):
        """
        Initialize GraphOnDemandAgent.

        Args:
            corpus_id: Optional corpus ID
            entity_k: Number of entities to retrieve
            community_k: Number of communities to retrieve
            rerank_k: Number of documents after reranking
        """
        super().__init__(name="graph_on_demand")
        self.corpus_id = corpus_id
        self.entity_k = entity_k
        self.community_k = community_k
        self.rerank_k = rerank_k

        # Initialize retrievers
        self.entity_retriever = create_entity_retriever(
            corpus_id=corpus_id,
            k=entity_k,
            expand_to_chunks=True
        )

        self.community_retriever = create_community_retriever(
            corpus_id=corpus_id,
            k=community_k,
            expand_to_chunks=True
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
            "GraphOnDemandAgent initialized",
            corpus_id=corpus_id,
            entity_k=entity_k,
            community_k=community_k
        )

    def build_graph(self) -> CompiledGraph:
        """
        Build the agent's LangGraph workflow.

        Returns:
            Compiled graph
        """
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("retrieve_entities", self._retrieve_entities_node)
        workflow.add_node("retrieve_communities", self._retrieve_communities_node)
        workflow.add_node("merge_and_rerank", self._merge_and_rerank_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Define edges
        workflow.set_entry_point("retrieve_entities")
        workflow.add_edge("retrieve_entities", "retrieve_communities")
        workflow.add_edge("retrieve_communities", "merge_and_rerank")
        workflow.add_edge("merge_and_rerank", "synthesize")
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    async def _retrieve_entities_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieve using entity search.

        Args:
            state: Current state

        Returns:
            State updates
        """
        query = state["query"]

        logger.info("Retrieving via entities", query_preview=query[:100])

        try:
            entity_docs = await self.entity_retriever.ainvoke(query)

            logger.info(f"Retrieved {len(entity_docs)} entity-related chunks")

            return {
                "entity_docs": entity_docs
            }

        except Exception as e:
            logger.error("Entity retrieval failed", error=str(e))
            return {"entity_docs": []}

    async def _retrieve_communities_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieve using community search.

        Args:
            state: Current state

        Returns:
            State updates
        """
        query = state["query"]

        logger.info("Retrieving via communities", query_preview=query[:100])

        try:
            community_docs = await self.community_retriever.ainvoke(query)

            logger.info(f"Retrieved {len(community_docs)} community-related chunks")

            return {
                "community_docs": community_docs
            }

        except Exception as e:
            logger.error("Community retrieval failed", error=str(e))
            return {"community_docs": []}

    async def _merge_and_rerank_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Merge entity and community results, then rerank.

        Args:
            state: Current state

        Returns:
            State updates
        """
        query = state["query"]
        entity_docs = state.get("entity_docs", [])
        community_docs = state.get("community_docs", [])

        # Merge documents (deduplicate by chunk_id)
        seen_chunks = set()
        merged_docs = []

        for doc in entity_docs + community_docs:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id and chunk_id in seen_chunks:
                continue

            if chunk_id:
                seen_chunks.add(chunk_id)

            merged_docs.append(doc)

        logger.info(f"Merged to {len(merged_docs)} unique documents")

        # Rerank
        if merged_docs:
            reranked_docs = await self.reranker.arerank(
                query=query,
                documents=merged_docs,
                top_k=self.rerank_k
            )

            logger.info(f"Reranked to top {len(reranked_docs)} documents")

            return {"reranked_docs": reranked_docs}
        else:
            return {"reranked_docs": []}

    async def _synthesize_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Synthesize answer from graph-retrieved context.

        Args:
            state: Current state

        Returns:
            State updates with answer
        """
        query = state["query"]
        docs = state.get("reranked_docs", [])

        if not docs:
            return {
                "answer": "I couldn't find relevant information in the knowledge graph to answer your question.",
                "citations": [],
                "confidence": 0.0
            }

        # Build context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            entity_info = f"Entity: {metadata.get('entity_name', 'N/A')} ({metadata.get('entity_type', 'N/A')})"
            community_info = f"Community: {metadata.get('community_name', 'N/A')}"

            context_parts.append(
                f"[{i}] {entity_info} | {community_info}\n{doc.page_content}"
            )

        context = "\n\n".join(context_parts)

        # Extract relationships
        relationships = self._extract_relationships(docs)

        # Generate answer
        prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)
        chain = prompt | self.llm | StrOutputParser()

        answer = await chain.ainvoke({
            "query": query,
            "context": context,
            "relationships": relationships
        })

        # Build citations
        citations = [
            Citation(
                citation_number=i,
                content=doc.page_content[:200],
                doc_id=doc.metadata.get("doc_id", "unknown"),
                metadata={
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "entity_name": doc.metadata.get("entity_name"),
                    "entity_type": doc.metadata.get("entity_type"),
                    "community_name": doc.metadata.get("community_name")
                }
            )
            for i, doc in enumerate(docs, 1)
        ]

        logger.info(
            "Synthesis completed",
            answer_length=len(answer),
            citations=len(citations)
        )

        return {
            "answer": answer,
            "citations": [c.dict() for c in citations],
            "confidence": 0.85 if len(docs) >= 3 else 0.7,
            "sources": docs
        }

    def _extract_relationships(self, docs: List[Document]) -> str:
        """
        Extract relationship information from document metadata.

        Args:
            docs: Retrieved documents

        Returns:
            Formatted relationship string
        """
        entities = set()
        communities = set()

        for doc in docs:
            entity_name = doc.metadata.get("entity_name")
            if entity_name:
                entities.add(f"{entity_name} ({doc.metadata.get('entity_type', 'N/A')})")

            community_name = doc.metadata.get("community_name")
            if community_name:
                communities.add(community_name)

        relationships = []

        if entities:
            relationships.append(f"Entities: {', '.join(list(entities)[:10])}")

        if communities:
            relationships.append(f"Communities: {', '.join(list(communities)[:5])}")

        return "\n".join(relationships) if relationships else "No explicit relationships found"


def create_graph_on_demand_agent(corpus_id: Optional[str] = None, **kwargs) -> GraphOnDemandAgent:
    """
    Create a GraphOnDemandAgent instance.

    Args:
        corpus_id: Optional corpus ID
        **kwargs: Additional arguments

    Returns:
        GraphOnDemandAgent
    """
    return GraphOnDemandAgent(corpus_id=corpus_id, **kwargs)
