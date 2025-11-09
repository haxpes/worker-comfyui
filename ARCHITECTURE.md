# AI-SME System Architecture

## Overview
Multi-agent AI consultant system with domain-aware reasoning, hybrid retrieval, and persistent memory.

## Technology Stack

### Core Framework
- **Backend**: FastAPI (async, lifespan management)
- **Orchestration**: LangGraph (primary - state management, workflows)
- **Components**: LangChain (~80% - retrievers, tools, LCEL)
- **Specialized**: LlamaIndex (~limited - PDF parsing, specialized indexes)
- **Database**: PostgreSQL (single unified backend)
- **Frontend**: React (streaming chat, evidence panels, trace viewer)

### Key Libraries
```
fastapi==0.109.0
langgraph==0.2.28
langchain==0.1.20
langchain-openai==0.0.5
langchain-community==0.0.20
llama-index==0.9.48
psycopg[binary,pool]==3.1.16
pgvector==0.2.4
sqlalchemy==2.0.25
sentence-transformers==2.3.1
rank-bm25==0.2.2
spacy==3.7.2
```

## System Architecture

### Component Hierarchy
```
User Request
    ↓
FastAPI Endpoint (SSE streaming)
    ↓
RefineQuestionAgent (pre-processing)
    ↓
Router (Mixture of Experts) ← Memory (long-term preferences)
    ↓
Agent Selection (single or parallel)
    ↓
Agent Workflows (LangGraph subgraphs)
    ├── Retrieval Tools (BaseRetriever + create_retriever_tool)
    ├── Processing Nodes (fusion, rerank, synthesis)
    └── Validation (Neuro-Symbolic Validator)
    ↓
CritiqueAgent (post-processing)
    ↓
Response Synthesis ← Memory (short-term context)
    ↓
Streaming Response + Citations
```

### LangGraph Architecture Pattern

#### Main Graph Structure
```python
# Hierarchical Teams Pattern
MainGraph:
  - RefineQuestionNode
  - RouterNode (MoE coordinator)
  - AgentTeamNode (manages agent subgraphs)
  - CritiqueNode
  - SynthesizeNode
```

#### Agent Subgraphs
Each agent is a compiled LangGraph workflow:
```python
AgentSubgraph (e.g., LeanHybridAgent):
  - RetrieveNode
  - RerankNode
  - SynthesizeNode
  - ValidateNode (optional)
```

#### State Management
```python
class MainState(MessagesState):
    """Shared state across all workflows"""
    # Query context
    refined_query: str
    query_facets: list[str]

    # Agent execution
    selected_agents: list[str]
    agent_results: dict[str, AgentResult]

    # Retrieved evidence
    retrieved_docs: list[Document]
    seen_doc_ids: set[str]

    # Synthesis
    answer: str
    citations: list[Citation]
    confidence: float

    # Memory
    thread_id: str
    user_id: str
    conversation_summary: str
```

## Database Schema

### Core Tables

#### Document Storage
```sql
-- Corpus metadata
CREATE TABLE corpora (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    file_path TEXT NOT NULL,  -- e.g., /storage/medical-standards/
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Documents (references to files)
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_id UUID REFERENCES corpora(id) ON DELETE CASCADE,
    filename VARCHAR(512) NOT NULL,
    file_path TEXT NOT NULL,
    title TEXT,
    doc_type VARCHAR(50),  -- pdf, markdown
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(corpus_id, filename)
);

-- Chunks (global searchable)
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    corpus_id UUID REFERENCES corpora(id),
    content TEXT NOT NULL,
    embedding vector(1536),
    chunk_index INTEGER NOT NULL,
    section_type VARCHAR(50),  -- headline, paragraph, list
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_chunks_content_tsv ON chunks USING gin(content_tsv);
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_corpus_id ON chunks(corpus_id);
```

#### RAPTOR (per-corpus)
```sql
-- RAPTOR summaries (hierarchical, corpus-specific)
CREATE TABLE raptor_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_id UUID REFERENCES corpora(id) ON DELETE CASCADE,
    level INTEGER NOT NULL,  -- 1, 2, 3
    content TEXT NOT NULL,
    embedding vector(1536),
    parent_id UUID REFERENCES raptor_summaries(id),
    topic VARCHAR(512),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- RAPTOR membership (summaries → chunks)
CREATE TABLE raptor_membership (
    summary_id UUID REFERENCES raptor_summaries(id) ON DELETE CASCADE,
    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    weight FLOAT DEFAULT 1.0,
    PRIMARY KEY (summary_id, chunk_id)
);

CREATE INDEX idx_raptor_summaries_corpus ON raptor_summaries(corpus_id, level);
CREATE INDEX idx_raptor_summaries_embedding ON raptor_summaries USING hnsw (embedding vector_cosine_ops);
```

#### GraphRAG (per-corpus)
```sql
-- Entities (corpus-specific)
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_id UUID REFERENCES corpora(id) ON DELETE CASCADE,
    name VARCHAR(512) NOT NULL,
    entity_type VARCHAR(100),  -- PERSON, ORG, STANDARD, etc.
    description TEXT,
    embedding vector(1536),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(corpus_id, name, entity_type)
);

-- Entity occurrences
CREATE TABLE entity_occurrences (
    entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    mention_text TEXT,
    position INTEGER,
    PRIMARY KEY (entity_id, chunk_id, position)
);

-- Relationships (corpus-specific graph)
CREATE TABLE relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_id UUID REFERENCES corpora(id) ON DELETE CASCADE,
    source_entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    relation_type VARCHAR(100),  -- co-mention, raptor-link, sequential
    weight FLOAT DEFAULT 1.0,
    metadata JSONB,
    UNIQUE(source_entity_id, target_entity_id, relation_type)
);

-- Communities (topic clusters, corpus-specific)
CREATE TABLE communities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_id UUID REFERENCES corpora(id) ON DELETE CASCADE,
    name VARCHAR(512),
    description TEXT,
    level INTEGER DEFAULT 0,  -- Louvain hierarchy level
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Community membership
CREATE TABLE community_members (
    community_id UUID REFERENCES communities(id) ON DELETE CASCADE,
    entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (community_id, entity_id)
);

CREATE INDEX idx_entities_corpus ON entities(corpus_id);
CREATE INDEX idx_entities_name ON entities(name);
CREATE INDEX idx_relationships_corpus ON relationships(corpus_id);
```

#### Experience Knowledge (per-corpus)
```sql
-- Best practices, lessons learned (corpus-specific)
CREATE TABLE experience_knowledge (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_id UUID REFERENCES corpora(id) ON DELETE CASCADE,
    category VARCHAR(100),  -- best_practice, risk, constraint, mitigation
    title VARCHAR(512),
    content TEXT NOT NULL,
    embedding vector(1536),
    source_chunks UUID[],  -- Array of chunk IDs
    confidence FLOAT DEFAULT 1.0,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_experience_corpus ON experience_knowledge(corpus_id, category);
CREATE INDEX idx_experience_embedding ON experience_knowledge USING hnsw (embedding vector_cosine_ops);
```

#### Memory System (LangGraph integration)
```sql
-- Short-term memory (PostgresSaver - thread-level)
-- Managed by LangGraph checkpointing system

-- Long-term memory (PostgresStore - cross-thread)
-- Managed by LangGraph store system

-- User preferences and patterns
CREATE TABLE user_preferences (
    user_id VARCHAR(255) PRIMARY KEY,
    preferences JSONB NOT NULL,
    query_patterns JSONB,  -- Learned patterns for router bias
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Conversation summaries (for context restoration)
CREATE TABLE conversation_summaries (
    thread_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) REFERENCES user_preferences(user_id),
    summary TEXT NOT NULL,
    embedding vector(1536),
    message_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_conversation_summaries_user ON conversation_summaries(user_id);
CREATE INDEX idx_conversation_summaries_embedding ON conversation_summaries USING hnsw (embedding vector_cosine_ops);
```

#### Analytics and Observability
```sql
-- Query logs
CREATE TABLE query_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255),
    thread_id VARCHAR(255),
    query TEXT NOT NULL,
    refined_query TEXT,
    selected_agents VARCHAR(100)[],
    execution_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Agent performance
CREATE TABLE agent_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name VARCHAR(100) NOT NULL,
    query_id UUID REFERENCES query_logs(id),
    execution_time_ms INTEGER,
    doc_count INTEGER,
    confidence FLOAT,
    success BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Validator scores
CREATE TABLE validator_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID REFERENCES query_logs(id),
    factuality FLOAT,
    consistency FLOAT,
    completeness FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_query_logs_user ON query_logs(user_id);
CREATE INDEX idx_query_logs_thread ON query_logs(thread_id);
CREATE INDEX idx_agent_performance_agent ON agent_performance(agent_name);
```

## Project Structure

```
ai-sme-system/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI app entry point
│   │   ├── config.py                  # Settings (Pydantic)
│   │   │
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── state.py              # LangGraph state definitions
│   │   │   ├── memory.py             # MemoryManager (PostgresSaver/Store)
│   │   │   └── observability.py     # Logging, tracing setup
│   │   │
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── connection.py         # Database connection pool
│   │   │   ├── models.py             # SQLAlchemy models
│   │   │   ├── migrations/           # Alembic migrations
│   │   │   │   └── versions/
│   │   │   └── init_db.sql           # Initial schema
│   │   │
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── pdf_parser.py         # LlamaIndex PDF extraction
│   │   │   ├── chunker.py            # Semantic chunking
│   │   │   ├── embedder.py           # OpenAI embeddings
│   │   │   ├── indexer.py            # Main indexing pipeline
│   │   │   ├── raptor_builder.py     # RAPTOR hierarchy
│   │   │   ├── entity_extractor.py   # spaCy + LLM NER
│   │   │   ├── graph_builder.py      # GraphRAG construction
│   │   │   └── experience_extractor.py # Best practices extraction
│   │   │
│   │   ├── retrieval/
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # Base retriever interface
│   │   │   ├── bm25.py               # PostgresBM25Retriever
│   │   │   ├── dense.py              # PostgresDenseRetriever
│   │   │   ├── hybrid.py             # PostgresHybridRetriever
│   │   │   ├── hyde.py               # HyDERetriever
│   │   │   ├── raptor.py             # RAPTORRetriever
│   │   │   ├── entity.py             # EntityRetriever
│   │   │   ├── community.py          # CommunityRetriever
│   │   │   ├── graph_traversal.py    # GraphTraversalRetriever
│   │   │   ├── experience.py         # ExperienceRetriever
│   │   │   ├── fusion.py             # RRF + MMR
│   │   │   ├── reranker.py           # Cross-encoder + LLM reranking
│   │   │   └── profiles.py           # DEEP/WIDE/EXPLORE profiles
│   │   │
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── registry.py           # Agent registry
│   │   │   ├── base.py               # Base agent interface
│   │   │   │
│   │   │   ├── core/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── lean_hybrid.py    # LeanHybridAgent
│   │   │   │   ├── raptor_agent.py   # RAPTORAgent
│   │   │   │   ├── plan_then_read.py # PlanThenReadAgent
│   │   │   │   ├── evidence_first.py # EvidenceFirstAgent
│   │   │   │   ├── graph_on_demand.py # GraphOnDemandAgent
│   │   │   │   ├── distill_first.py  # DistillFirstAgent
│   │   │   │   ├── hrm.py            # HRMAgent
│   │   │   │   ├── cot.py            # CoTAgent
│   │   │   │   ├── react.py          # ReActAgent
│   │   │   │   └── self_rag.py       # SelfRAGAgent
│   │   │   │
│   │   │   └── extended/
│   │   │       ├── __init__.py
│   │   │       ├── refine_question.py # RefineQuestionAgent
│   │   │       ├── critique.py       # CritiqueAgent
│   │   │       └── whatif.py         # WhatIfAgent
│   │   │
│   │   ├── orchestration/
│   │   │   ├── __init__.py
│   │   │   ├── router.py             # Router (MoE) implementation
│   │   │   ├── main_graph.py         # Main LangGraph workflow
│   │   │   ├── synthesis.py          # Multi-agent synthesis
│   │   │   └── validator.py          # Neuro-Symbolic Validator
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py               # Chat endpoints (streaming)
│   │   │   ├── agents.py             # Agent management
│   │   │   ├── corpus.py             # Corpus management
│   │   │   ├── storage.py            # Storage browser
│   │   │   └── memory.py             # Memory queries
│   │   │
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── logger.py             # Structured logging
│   │       ├── prompts.py            # Prompt templates
│   │       ├── metrics.py            # OpenTelemetry metrics
│   │       └── helpers.py            # Common utilities
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── unit/
│   │   ├── integration/
│   │   └── fixtures/
│   │
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── .env.example
│
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.jsx
│   │   │
│   │   ├── components/
│   │   │   ├── chat/
│   │   │   │   ├── ChatInterface.jsx
│   │   │   │   ├── MessageList.jsx
│   │   │   │   ├── MessageInput.jsx
│   │   │   │   └── StreamingIndicator.jsx
│   │   │   │
│   │   │   ├── evidence/
│   │   │   │   ├── EvidencePanel.jsx
│   │   │   │   ├── CitationCard.jsx
│   │   │   │   └── ConfidenceScore.jsx
│   │   │   │
│   │   │   ├── trace/
│   │   │   │   ├── TraceViewer.jsx
│   │   │   │   ├── GraphVisualizer.jsx
│   │   │   │   └── NodeDetails.jsx
│   │   │   │
│   │   │   ├── storage/
│   │   │   │   ├── StorageBrowser.jsx
│   │   │   │   ├── CorpusExplorer.jsx
│   │   │   │   └── DocumentViewer.jsx
│   │   │   │
│   │   │   └── common/
│   │   │       ├── Button.jsx
│   │   │       ├── Card.jsx
│   │   │       └── Loading.jsx
│   │   │
│   │   ├── services/
│   │   │   ├── api.js               # API client
│   │   │   └── websocket.js         # SSE handling
│   │   │
│   │   └── styles/
│   │       └── main.css
│   │
│   └── public/
│       └── index.html
│
├── storage/
│   ├── medical-standards/           # Example corpus
│   ├── software-regulations/        # Example corpus
│   └── general-knowledge/           # Example corpus
│
├── logs/
│   └── app.log
│
├── scripts/
│   ├── setup.sh                     # Initial setup
│   └── test_env.sh                  # Environment validation
│
├── docker-compose.yml               # PostgreSQL + pgvector
├── cli.py                           # Index management CLI
├── README.md
├── ARCHITECTURE.md                  # This file
└── .env.example
```

## Retrieval Architecture

### Retriever Hierarchy
All retrievers inherit from `langchain.schema.BaseRetriever`:

```python
from langchain.schema import BaseRetriever
from langchain.tools import create_retriever_tool

class PostgresBM25Retriever(BaseRetriever):
    """Full-text search using PostgreSQL"""
    # Global search by default, optional corpus filter

class PostgresDenseRetriever(BaseRetriever):
    """Vector similarity using pgvector"""
    # Global search by default, optional corpus filter

class RAPTORRetriever(BaseRetriever):
    """Hierarchical summary retrieval"""
    # Corpus-specific (operates on per-corpus RAPTOR trees)

# Wrap for LangGraph tool use
bm25_tool = create_retriever_tool(
    PostgresBM25Retriever(),
    name="bm25_search",
    description="Full-text keyword search"
)
```

### Retrieval Profiles

```python
class RetrievalProfile(Enum):
    DEEP = "deep"          # Exhaustive, high recall
    WIDE = "wide"          # Broad coverage, diverse
    EXPLORE = "explore"    # Novelty-focused, new info

def get_retrieval_params(profile: RetrievalProfile) -> dict:
    if profile == RetrievalProfile.DEEP:
        return {
            "bm25_k": 30,
            "dense_k": 30,
            "fusion_k": 20,
            "mmr_lambda": 0.5  # Less diversity
        }
    elif profile == RetrievalProfile.WIDE:
        return {
            "bm25_k": 40,
            "dense_k": 40,
            "fusion_k": 25,
            "mmr_lambda": 0.8  # High diversity
        }
    # ... etc
```

## Agent Workflows

### LeanHybridAgent (Example)
```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

def create_lean_hybrid_agent() -> CompiledGraph:
    workflow = StateGraph(AgentState)

    # Define nodes
    async def retrieve_node(state: AgentState) -> AgentState:
        """Hybrid retrieval: BM25 + Dense"""
        query = state["refined_query"]

        # Parallel retrieval
        bm25_docs = await bm25_retriever.aget_relevant_documents(query)
        dense_docs = await dense_retriever.aget_relevant_documents(query)

        # RRF fusion
        fused_docs = rrf_fusion(bm25_docs, dense_docs, k=config.FUSION_K)

        # MMR diversity
        diverse_docs = mmr_rerank(fused_docs, lambda_param=config.MMR_LAMBDA)

        return {"retrieved_docs": diverse_docs}

    async def rerank_node(state: AgentState) -> AgentState:
        """Cross-encoder reranking"""
        docs = state["retrieved_docs"]
        query = state["refined_query"]

        reranked = await cross_encoder_rerank(query, docs, top_k=config.CITATIONS_COUNT)

        return {"retrieved_docs": reranked}

    async def synthesize_node(state: AgentState) -> AgentState:
        """Generate answer with citations"""
        docs = state["retrieved_docs"]
        query = state["refined_query"]

        # Build context
        context = "\n\n".join([f"[{i}] {doc.page_content}" for i, doc in enumerate(docs)])

        # LLM synthesis
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYNTHESIS_PROMPT),
            ("user", f"Query: {query}\n\nContext:\n{context}")
        ])

        chain = prompt | llm | StrOutputParser()
        answer = await chain.ainvoke({})

        # Extract citations
        citations = [{"doc_id": doc.metadata["id"], "content": doc.page_content}
                     for doc in docs]

        return {
            "answer": answer,
            "citations": citations,
            "confidence": 0.8  # Placeholder
        }

    # Build graph
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("synthesize", synthesize_node)

    # Define flow
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow.compile()
```

### Router (Mixture of Experts)
```python
from langgraph.types import Command

async def router_node(state: MainState) -> Command:
    """Select appropriate agent(s) based on query analysis"""
    query = state["refined_query"]
    facets = state["query_facets"]

    # Load user preferences from long-term memory
    user_patterns = await memory_manager.get_query_patterns(state["user_id"])

    # Analyze query
    analysis = await analyze_query(query, facets, user_patterns)

    # Route decision
    if analysis["complexity"] == "simple":
        selected = ["lean_hybrid"]
    elif analysis["intent"] == "compliance":
        selected = ["evidence_first"]
    elif analysis["requires_relationships"]:
        selected = ["graph_on_demand"]
    elif analysis["complexity"] == "multi_part":
        selected = ["plan_then_read"]
    else:
        selected = ["lean_hybrid"]  # Default

    # Log decision
    logger.info(f"Router selected: {selected} for query: {query[:50]}...")

    # Update state and route using Command
    return Command(
        update={"selected_agents": selected},
        goto="agent_team"
    )
```

## Memory System

### MemoryManager (Lifespan Integration)
```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

class MemoryManager:
    """Unified memory management for LangGraph"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.checkpointer = None  # PostgresSaver
        self.store = None         # PostgresStore

    async def initialize(self):
        """Initialize memory components"""
        # Short-term memory (thread-level)
        self.checkpointer = PostgresSaver.from_conn_string(self.database_url)
        await self.checkpointer.setup()

        # Long-term memory (cross-thread)
        self.store = PostgresStore.from_conn_string(self.database_url)
        await self.store.setup()

    async def shutdown(self):
        """Cleanup connections"""
        if self.checkpointer:
            await self.checkpointer.close()
        if self.store:
            await self.store.close()

    async def save_conversation_summary(self, thread_id: str, summary: str):
        """Persist conversation summary with embedding"""
        embedding = await get_embedding(summary)
        # Store in PostgresStore under user namespace
        # Also update conversation_summaries table for semantic search

    async def get_query_patterns(self, user_id: str) -> dict:
        """Retrieve learned query patterns for router bias"""
        items = await self.store.search(
            namespace=("users", user_id, "query_patterns"),
            limit=10
        )
        return items
```

### FastAPI Lifespan Integration
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Initializing AI-SME System...")

    # Initialize memory
    memory_manager = MemoryManager(settings.DATABASE_URL)
    await memory_manager.initialize()
    app.state.memory = memory_manager

    # Initialize retrievers
    app.state.retrievers = await initialize_retrievers()

    # Initialize agents
    app.state.agents = await initialize_agents()

    # Initialize router
    app.state.router = create_router_graph(memory_manager.checkpointer)

    logger.info("System ready!")

    yield

    # Shutdown
    logger.info("Shutting down...")
    await memory_manager.shutdown()

app = FastAPI(lifespan=lifespan)
```

## Streaming Architecture

### SSE Streaming
```python
from fastapi.responses import StreamingResponse

@router.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """Stream chat responses using Server-Sent Events"""

    async def event_generator():
        # Get router graph with memory
        graph = request.app.state.router

        # Configure streaming
        config = {
            "configurable": {
                "thread_id": request.thread_id,
                "checkpoint_ns": request.user_id
            }
        }

        # Stream events
        async for event in graph.astream(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
            stream_mode="values"  # Stream state updates
        ):
            # Format as SSE
            if "answer" in event:
                yield f"data: {json.dumps({'type': 'token', 'content': event['answer']})}\n\n"
            elif "retrieved_docs" in event:
                yield f"data: {json.dumps({'type': 'docs', 'count': len(event['retrieved_docs'])})}\n\n"
            elif "selected_agents" in event:
                yield f"data: {json.dumps({'type': 'agents', 'agents': event['selected_agents']})}\n\n"

        # Final event
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
```

## Configuration Management

### Environment Variables
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str

    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/ai_sme"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10

    # Storage
    CORPUS_BASE_PATH: str = "/storage"

    # Retrieval Parameters
    BM25_K: int = 20
    DENSE_K: int = 20
    FUSION_K: int = 10
    MMR_K: int = 5
    MMR_LAMBDA: float = 0.7
    CITATIONS_COUNT: int = 3

    # LLM
    DEFAULT_MODEL: str = "gpt-4o-mini"
    COMPLEX_MODEL: str = "gpt-4o"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Reranking
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Validation
    VALIDATOR_THRESHOLD: float = 0.7

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"

    # Observability
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "ai-sme-system"

    class Config:
        env_file = ".env"

settings = Settings()
```

## Implementation Phases

### Phase 1: Foundation (Weeks 1-4)
**Goal**: Working system with basic retrieval and single agent

**Deliverables**:
- ✅ PostgreSQL setup with pgvector, pg_search
- ✅ Basic schema (documents, chunks, indexes)
- ✅ File storage structure
- ✅ PDF ingestion pipeline
- ✅ Semantic chunking
- ✅ BM25 + Dense retrievers
- ✅ RRF fusion, MMR diversity
- ✅ Cross-encoder reranking
- ✅ LeanHybridAgent (LangGraph)
- ✅ FastAPI endpoints (streaming)
- ✅ Basic React chat UI
- ✅ Logging system
- ✅ CLI for indexing

### Phase 2: Core Agents (Weeks 5-8)
**Goal**: Multi-agent system with memory and routing

**Deliverables**:
- ✅ RAPTOR pipeline (hierarchical summaries)
- ✅ RAPTORAgent
- ✅ PlanThenReadAgent (query decomposition)
- ✅ EvidenceFirstAgent (extractive citations)
- ✅ Router (MoE) implementation
- ✅ Memory system (PostgresSaver, PostgresStore)
- ✅ RefineQuestionAgent (pre-processing)
- ✅ Evidence panel in UI
- ✅ Agent selection UI

### Phase 3: Advanced Features (Weeks 9-12)
**Goal**: GraphRAG, self-reflection, validation

**Deliverables**:
- ✅ Entity extraction (spaCy + LLM)
- ✅ Community detection (Louvain)
- ✅ Graph construction (co-mentions, RAPTOR links)
- ✅ GraphOnDemandAgent (ephemeral subgraphs)
- ✅ EntityRetriever, CommunityRetriever, GraphTraversalRetriever
- ✅ SelfRAGAgent (reflection, iteration)
- ✅ Advanced Synthesis (multi-perspective)
- ✅ Neuro-Symbolic Validator
- ✅ CritiqueAgent (post-processing)
- ✅ Trace viewer UI
- ✅ Storage browser UI
- ✅ Performance optimization

### Phase 4: Production Ready (Weeks 13-16)
**Goal**: Complete agent suite, testing, deployment

**Deliverables**:
- ✅ DistillFirstAgent
- ✅ HRMAgent (hierarchical reasoning)
- ✅ CoTAgent (chain-of-thought)
- ✅ ReActAgent (tool-using)
- ✅ WhatIfAgent (scenario analysis)
- ✅ Agentic Context Engineering (ACE)
- ✅ Experience knowledge extraction
- ✅ ExperienceRetriever
- ✅ Comprehensive test suite
- ✅ OpenTelemetry integration
- ✅ LangSmith tracing
- ✅ Deployment guides
- ✅ Performance tuning
- ✅ Load testing
- ✅ Documentation

## Design Principles

### KISS (Keep It Simple, Stupid)
- Start with simplest solutions
- Single corpus folder in Phase 1
- Add complexity only when needed
- Clear, readable code over clever optimizations

### DRY (Don't Repeat Yourself)
- Shared base classes for retrievers and agents
- Common utilities and helpers
- Reusable prompt templates
- Centralized configuration

### YAGNI (You Aren't Gonna Need It)
- No premature optimization
- Build features when required, not "just in case"
- No tests in Phase 1 (manual testing only)
- Simple architecture, evolve as needed

## Next Steps
1. Create project structure
2. Implement Phase 1 foundation
3. Iterate through phases 2-4
4. Test and optimize
5. Deploy and document
