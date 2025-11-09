# AI-SME System Implementation Roadmap

## Overview
This document outlines the detailed implementation plan across 4 phases, with concrete tasks, dependencies, and deliverables.

---

## Phase 1: Foundation (Weeks 1-4)

### Goal
Build a working system with single agent, hybrid retrieval, and streaming chat interface.

### Tasks

#### 1.1 Project Setup (Week 1, Days 1-2)
- [ ] Create project directory structure
- [ ] Initialize git repository
- [ ] Create requirements.txt with dependencies
- [ ] Setup .env.example with all configuration
- [ ] Create docker-compose.yml for PostgreSQL
- [ ] Write setup scripts (setup.sh, test_env.sh)
- [ ] Initialize README with quickstart

**Dependencies**: None
**Deliverable**: Clean project structure ready for development

#### 1.2 Database Layer (Week 1, Days 3-5)
- [ ] Create init_db.sql with schema for Phase 1
  - corpora table
  - documents table
  - chunks table (with vector and tsvector)
  - Indexes: HNSW for vectors, GIN for full-text
- [ ] Implement db/connection.py (connection pooling)
- [ ] Implement db/models.py (SQLAlchemy ORM)
- [ ] Create Alembic migration setup
- [ ] Test database connectivity and extensions

**Dependencies**: 1.1
**Deliverable**: PostgreSQL database with pgvector and full-text search ready

#### 1.3 Configuration and Logging (Week 1, Days 6-7)
- [ ] Implement config.py (Pydantic Settings)
- [ ] Implement utils/logger.py (structured logging)
  - Console handler with colors
  - File handler (logs/app.log)
  - Reset on app restart
  - JSON structured format
- [ ] Create core/observability.py skeleton
- [ ] Test configuration loading from .env

**Dependencies**: 1.1
**Deliverable**: Configuration and logging infrastructure

#### 1.4 Document Ingestion Pipeline (Week 2, Days 1-4)
- [ ] Implement ingestion/pdf_parser.py
  - Use LlamaIndex UnstructuredReader
  - Extract headlines and paragraphs only
  - Preserve structure metadata
- [ ] Implement ingestion/chunker.py
  - Semantic chunking (LlamaIndex SemanticSplitterNodeParser)
  - No max chunk size, no overlaps
  - Section classification (headline, paragraph, list)
- [ ] Implement ingestion/embedder.py
  - OpenAI text-embedding-3-small
  - Batch processing with retry logic
- [ ] Implement ingestion/indexer.py
  - Main pipeline: parse → chunk → embed → store
  - Corpus folder scanning
  - Progress tracking
- [ ] Create CLI command: python cli.py index <corpus_path>

**Dependencies**: 1.2, 1.3
**Deliverable**: Working ingestion pipeline

#### 1.5 Retrieval System (Week 2, Days 5-7 + Week 3, Days 1-2)
- [ ] Implement retrieval/base.py
  - Base retriever interface
  - Common utilities
- [ ] Implement retrieval/bm25.py (PostgresBM25Retriever)
  - Full-text search using ts_rank
  - Configurable k parameter
  - Return LangChain Document objects
- [ ] Implement retrieval/dense.py (PostgresDenseRetriever)
  - Vector similarity using pgvector
  - Cosine distance
  - Configurable k parameter
- [ ] Implement retrieval/hybrid.py (PostgresHybridRetriever)
  - Combine BM25 + Dense
  - Use LangChain EnsembleRetriever
- [ ] Implement retrieval/fusion.py
  - RRF (Reciprocal Rank Fusion)
  - MMR (Maximum Marginal Relevance)
  - Configurable parameters
- [ ] Implement retrieval/reranker.py
  - Cross-encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2)
  - Local inference (sentence-transformers)
  - Batch processing
- [ ] Test retrievers with sample corpus

**Dependencies**: 1.2, 1.3, 1.4
**Deliverable**: Hybrid retrieval system with reranking

#### 1.6 LeanHybridAgent (Week 3, Days 3-5)
- [ ] Implement core/state.py
  - AgentState TypedDict
  - MainState extending MessagesState
- [ ] Implement utils/prompts.py
  - Hardcoded prompts for synthesis
  - Citation instructions
- [ ] Implement agents/base.py
  - Base agent interface
  - Common utilities
- [ ] Implement agents/core/lean_hybrid.py
  - Create LangGraph StateGraph
  - Nodes: retrieve, rerank, synthesize
  - Use hybrid retriever + RRF + MMR + cross-encoder
  - LLM synthesis with GPT-4o-mini
  - Extract citations from results
  - Compile to CompiledGraph
- [ ] Test agent end-to-end

**Dependencies**: 1.5
**Deliverable**: Working LeanHybridAgent

#### 1.7 FastAPI Backend (Week 3, Days 6-7 + Week 4, Days 1-2)
- [ ] Implement main.py
  - FastAPI app with lifespan
  - CORS middleware
  - Error handling
- [ ] Implement api/chat.py
  - POST /chat/stream (SSE streaming)
  - Request/Response models
  - Integration with LeanHybridAgent
  - Stream state updates as events
- [ ] Implement api/agents.py
  - GET /agents/list
- [ ] Test endpoints with curl/Postman
- [ ] Add health check endpoint

**Dependencies**: 1.6
**Deliverable**: FastAPI backend with streaming

#### 1.8 React Frontend (Week 4, Days 3-6)
- [ ] Initialize React project (Vite)
- [ ] Implement src/services/api.js
  - EventSource for SSE
  - API client functions
- [ ] Implement src/components/chat/ChatInterface.jsx
  - Message input
  - Message list
  - Streaming indicator
- [ ] Implement src/components/chat/MessageList.jsx
  - Display messages
  - Show citations
- [ ] Implement src/components/evidence/CitationCard.jsx
  - Show doc snippet
  - Confidence score (placeholder)
- [ ] Implement basic styling
- [ ] Test frontend with backend

**Dependencies**: 1.7
**Deliverable**: Working chat interface

#### 1.9 Integration and Documentation (Week 4, Day 7)
- [ ] End-to-end testing (manual)
- [ ] Update README with:
  - Setup instructions
  - How to run
  - How to index corpus
  - How to use chat
- [ ] Create sample corpus (2-3 PDFs)
- [ ] Record demo video/screenshots

**Dependencies**: 1.8
**Deliverable**: Complete Phase 1 system

---

## Phase 2: Core Agents (Weeks 5-8)

### Goal
Multi-agent system with memory, routing, and RAPTOR retrieval.

### Tasks

#### 2.1 Database Schema Expansion (Week 5, Days 1-2)
- [ ] Add RAPTOR tables (summaries, membership)
- [ ] Add memory tables (user_preferences, conversation_summaries)
- [ ] Add LangGraph checkpointer tables
- [ ] Add LangGraph store tables
- [ ] Create migration script
- [ ] Test schema changes

**Dependencies**: Phase 1 complete
**Deliverable**: Expanded database schema

#### 2.2 RAPTOR Pipeline (Week 5, Days 3-7)
- [ ] Implement ingestion/raptor_builder.py
  - Hierarchical clustering (RAPTOR algorithm)
  - L1: Chunk clusters with summaries
  - L2: Summary clusters with summaries
  - L3: Corpus-level summary
  - Store summaries with embeddings
  - Track membership relationships
- [ ] Implement retrieval/raptor.py (RAPTORRetriever)
  - Search summaries by similarity
  - Expand to underlying chunks via membership
  - Multi-level retrieval
- [ ] Add RAPTOR build to CLI: python cli.py build-raptor <corpus_id>
- [ ] Test RAPTOR on sample corpus

**Dependencies**: 2.1
**Deliverable**: RAPTOR hierarchical summaries

#### 2.3 Memory System (Week 6, Days 1-3)
- [ ] Implement core/memory.py (MemoryManager)
  - PostgresSaver initialization
  - PostgresStore initialization
  - Lifespan management
  - Connection pooling
- [ ] Integrate MemoryManager into main.py lifespan
- [ ] Implement conversation summary generation
  - Periodic summarization during chat
  - Embed and store summaries
- [ ] Implement query pattern learning
  - Track successful agent selections
  - Store in PostgresStore under user namespace
- [ ] Test memory persistence across sessions

**Dependencies**: 2.1
**Deliverable**: Persistent memory system

#### 2.4 Additional Agents (Week 6, Days 4-7 + Week 7, Days 1-3)
- [ ] Implement agents/core/raptor_agent.py
  - Use RAPTORRetriever
  - Multi-level context assembly
  - Hierarchy-aware synthesis
- [ ] Implement agents/core/plan_then_read.py
  - Query decomposition node
  - Per-subquery retrieval
  - Sub-answer synthesis
  - Final aggregation
- [ ] Implement agents/core/evidence_first.py
  - Extractive span extraction
  - LLMChainExtractor for compression
  - Strict citation mapping
  - Audit-friendly output
- [ ] Implement agents/extended/refine_question.py
  - Query disambiguation
  - Topical facet generation
  - Use long-term memory + RAPTOR L3
  - Suggest agents
- [ ] Implement agents/extended/critique.py
  - Completeness checking
  - Alternative perspectives
  - Contradiction detection
  - Quality scoring
- [ ] Test each agent individually

**Dependencies**: 2.2, 2.3
**Deliverable**: 5 working agents

#### 2.5 Router Implementation (Week 7, Days 4-7 + Week 8, Days 1-2)
- [ ] Implement orchestration/router.py
  - Query analysis function
  - Routing logic (complexity, intent, domain)
  - Integration with memory (query patterns)
  - Command-based state transitions
- [ ] Implement orchestration/main_graph.py
  - Main LangGraph workflow
  - Nodes: RefineQuestionNode, RouterNode, AgentTeamNode, CritiqueNode
  - State management
  - Checkpointing integration
- [ ] Implement orchestration/synthesis.py
  - Multi-agent result merging
  - Conflict resolution
  - Citation deduplication
- [ ] Implement agents/registry.py
  - Agent registration
  - Dynamic agent loading
- [ ] Test routing decisions

**Dependencies**: 2.4
**Deliverable**: Router (MoE) system

#### 2.6 UI Enhancements (Week 8, Days 3-6)
- [ ] Add agent selection dropdown to ChatInterface
  - Auto-route mode (default)
  - Manual agent selection
- [ ] Implement src/components/evidence/EvidencePanel.jsx
  - Show all citations
  - Document metadata
  - Confidence scores
  - Source links
- [ ] Add processing stages indicator
  - Show current node
  - Agent selection
  - Retrieval progress
- [ ] Improve styling and UX
- [ ] Test UI with multiple agents

**Dependencies**: 2.5
**Deliverable**: Enhanced UI

#### 2.7 Integration and Testing (Week 8, Day 7)
- [ ] End-to-end testing with all agents
- [ ] Test memory persistence
- [ ] Test router decisions
- [ ] Update documentation
- [ ] Performance profiling

**Dependencies**: 2.6
**Deliverable**: Complete Phase 2 system

---

## Phase 3: Advanced Features (Weeks 9-12)

### Goal
GraphRAG, self-reflection, validation, and advanced UI.

### Tasks

#### 3.1 Database Schema for GraphRAG (Week 9, Days 1-2)
- [ ] Add entities table
- [ ] Add entity_occurrences table
- [ ] Add relationships table
- [ ] Add communities table
- [ ] Add community_members table
- [ ] Create indexes
- [ ] Migration script

**Dependencies**: Phase 2 complete
**Deliverable**: GraphRAG schema

#### 3.2 Entity Extraction (Week 9, Days 3-5)
- [ ] Implement ingestion/entity_extractor.py
  - spaCy NER (en_core_web_lg)
  - LLM enhancement for descriptions
  - Custom entity types (STANDARD, REGULATION, ARTIFACT)
  - Store entities with embeddings
  - Track occurrences in chunks
- [ ] Add entity extraction to indexing pipeline
- [ ] Test on sample corpus

**Dependencies**: 3.1
**Deliverable**: Entity extraction

#### 3.3 Graph Construction (Week 9, Days 6-7 + Week 10, Days 1-2)
- [ ] Implement ingestion/graph_builder.py
  - Co-mention edges (entities in same chunk)
  - RAPTOR membership edges
  - Sequential adjacency edges
  - Weight calculation
  - Community detection (Louvain algorithm)
  - Community summarization
  - Store graph in PostgreSQL
- [ ] Add graph build to CLI: python cli.py build-graph <corpus_id>
- [ ] Test graph construction

**Dependencies**: 3.2
**Deliverable**: Knowledge graph construction

#### 3.4 GraphRAG Retrievers (Week 10, Days 3-5)
- [ ] Implement retrieval/entity.py (EntityRetriever)
  - Entity name search
  - Entity type filtering
  - Expand to related chunks
- [ ] Implement retrieval/community.py (CommunityRetriever)
  - Topic cluster search
  - Community summary retrieval
  - Member expansion
- [ ] Implement retrieval/graph_traversal.py (GraphTraversalRetriever)
  - Starting node identification
  - BFS/DFS traversal (1-2 hops)
  - Context packing from traversal
  - Ephemeral subgraph extraction
- [ ] Test retrievers

**Dependencies**: 3.3
**Deliverable**: GraphRAG retrievers

#### 3.5 GraphOnDemandAgent (Week 10, Days 6-7)
- [ ] Implement agents/core/graph_on_demand.py
  - Query entity extraction
  - Ephemeral subgraph construction
  - Graph traversal
  - Context assembly
  - Relation-aware synthesis
  - Output: answer + relation trace
- [ ] Test agent with relationship queries

**Dependencies**: 3.4
**Deliverable**: GraphOnDemandAgent

#### 3.6 SelfRAGAgent (Week 11, Days 1-3)
- [ ] Implement retrieval/profiles.py
  - DEEP, WIDE, EXPLORE profile definitions
  - Parameter sets for each profile
- [ ] Implement agents/core/self_rag.py
  - Initial retrieval (WIDE)
  - Reflection node (gap detection)
  - Iteration logic (wide → deep → explore)
  - Novelty filtering (seen_doc_ids)
  - Confidence evaluation
  - Convergence check (max 4 iterations)
  - Final synthesis with coverage analysis
- [ ] Test self-correction behavior

**Dependencies**: 3.5
**Deliverable**: SelfRAGAgent

#### 3.7 Advanced Synthesis (Week 11, Days 4-5)
- [ ] Enhance orchestration/synthesis.py
  - Multi-perspective summarization
  - Confidence computation
  - Uncertainty analysis
  - Alternative views
  - Contradiction highlighting
- [ ] Test with conflicting sources

**Dependencies**: 3.6
**Deliverable**: Advanced synthesis

#### 3.8 Neuro-Symbolic Validator (Week 11, Days 6-7)
- [ ] Implement orchestration/validator.py
  - Factuality checking
  - Consistency scoring
  - Completeness evaluation
  - Claim-level verification
  - Citation integrity check
  - Re-retrieval for low confidence claims
  - Configurable thresholds
- [ ] Integrate validator into agent workflows
- [ ] Test validation accuracy

**Dependencies**: 3.7
**Deliverable**: Validator system

#### 3.9 Advanced UI Features (Week 12, Days 1-4)
- [ ] Implement src/components/trace/TraceViewer.jsx
  - Graph workflow visualization
  - Node execution timeline
  - State snapshots
- [ ] Implement src/components/trace/GraphVisualizer.jsx
  - D3.js graph rendering
  - Entity relationship display
- [ ] Implement src/components/storage/StorageBrowser.jsx
  - Corpus explorer
  - Document viewer
  - Chunk browser
  - RAPTOR tree viewer
  - Graph viewer
- [ ] Implement search and filtering
- [ ] Polish UI/UX

**Dependencies**: 3.8
**Deliverable**: Complete UI

#### 3.10 Performance Optimization (Week 12, Days 5-7)
- [ ] Profile bottlenecks
- [ ] Optimize database queries
  - Index tuning
  - Query plan analysis
- [ ] Implement caching
  - HyDE embeddings
  - QFSD results
  - Redis integration (optional)
- [ ] Optimize embedding batching
- [ ] Parallel retrieval optimization
- [ ] Test performance improvements

**Dependencies**: 3.9
**Deliverable**: Optimized system

---

## Phase 4: Production Ready (Weeks 13-16)

### Goal
Complete agent suite, testing, monitoring, deployment.

### Tasks

#### 4.1 Remaining Core Agents (Week 13)
- [ ] Implement agents/core/distill_first.py
  - Distilled knowledge retrieval
  - Fallback to corpus if confidence < threshold
  - Conflict resolution
- [ ] Implement agents/core/hrm.py
  - Strategy/execution loop alternation
  - Hierarchical convergence
  - Decision matrix output
- [ ] Implement agents/core/cot.py
  - Step-by-step reasoning
  - Periodic evidence refresh
  - Step linking to sources
- [ ] Implement agents/core/react.py
  - Thought → Action → Observation loop
  - ToolNode integration
  - Recursion limit
  - Tool execution log
- [ ] Test all agents

**Dependencies**: Phase 3 complete
**Deliverable**: Complete agent catalog

#### 4.2 Extended Agents (Week 13)
- [ ] Implement agents/extended/whatif.py
  - Scenario parameterization
  - Counterfactual reasoning
  - Risk analysis
  - Mitigation suggestions
- [ ] Test WhatIfAgent

**Dependencies**: 4.1
**Deliverable**: Extended agents

#### 4.3 Experience Knowledge (Week 14, Days 1-3)
- [ ] Implement ingestion/experience_extractor.py
  - Best practices extraction
  - Risk identification
  - Constraint detection
  - Mitigation strategies
  - Store with embeddings
- [ ] Implement retrieval/experience.py (ExperienceRetriever)
  - Category filtering
  - Similarity search
  - Source tracing
- [ ] Integrate into DistillFirstAgent
- [ ] Add to CLI: python cli.py extract-experience <corpus_id>

**Dependencies**: 4.2
**Deliverable**: Experience knowledge system

#### 4.4 Agentic Context Engineering (ACE) (Week 14, Days 4-5)
- [ ] Implement orchestration/ace.py
  - Pre-retrieval meta-loop
  - Context enrichment
  - Iteration control
  - Sufficiency evaluation
- [ ] Integrate into main graph
- [ ] Test context enhancement

**Dependencies**: 4.3
**Deliverable**: ACE system

#### 4.5 Testing Infrastructure (Week 14, Days 6-7 + Week 15, Days 1-2)
- [ ] Setup pytest
- [ ] Create test fixtures
  - Sample corpora
  - Mock LLM responses
  - Test databases
- [ ] Unit tests
  - Retrievers
  - Fusion/rerank
  - QFSD generation
  - Entity extraction
  - Memory operations
- [ ] Integration tests
  - Each agent end-to-end
  - Router selection
  - Multi-agent synthesis
  - Citation integrity
  - Memory persistence
- [ ] Golden-answer tests
  - Reference Q&A pairs
  - Expected outputs
  - Threshold validation
- [ ] Run test suite

**Dependencies**: 4.4
**Deliverable**: Comprehensive tests

#### 4.6 Observability (Week 15, Days 3-5)
- [ ] Implement utils/metrics.py
  - OpenTelemetry setup
  - Custom metrics (retrieval counts, latency, etc.)
  - Histograms for validator scores
  - Router decision counters
- [ ] Integrate LangSmith tracing
  - Trace all LLM calls
  - Trace agent workflows
  - Custom tags and metadata
- [ ] Structured logging enhancements
  - Request IDs
  - User IDs
  - Thread IDs
  - Performance markers
- [ ] Test observability pipeline

**Dependencies**: 4.5
**Deliverable**: Full observability

#### 4.7 Deployment (Week 15, Days 6-7 + Week 16, Days 1-2)
- [ ] Create production Dockerfile
  - Multi-stage build
  - Dependency optimization
- [ ] Enhance docker-compose.yml
  - PostgreSQL
  - Redis (optional)
  - Application
  - Reverse proxy (nginx)
- [ ] Create deployment guide
  - Local deployment
  - Docker deployment
  - Cloud deployment (AWS/GCP/Azure)
- [ ] Environment configuration guide
- [ ] Security hardening
  - Environment variable management
  - API authentication
  - Rate limiting
  - Input validation
- [ ] Test deployment

**Dependencies**: 4.6
**Deliverable**: Production deployment

#### 4.8 Documentation (Week 16, Days 3-5)
- [ ] API documentation (OpenAPI/Swagger)
- [ ] Architecture diagrams (C4 model)
  - Context diagram
  - Container diagram
  - Component diagram
  - Data flow diagram
- [ ] User guide
  - Getting started
  - Corpus management
  - Using different agents
  - Understanding results
- [ ] Developer guide
  - Project structure
  - Adding new agents
  - Adding new retrievers
  - Extending the system
- [ ] Configuration reference
- [ ] Troubleshooting guide
- [ ] FAQ

**Dependencies**: 4.7
**Deliverable**: Complete documentation

#### 4.9 Performance Tuning and Load Testing (Week 16, Days 6-7)
- [ ] Load testing with Locust/k6
  - Concurrent users (100+)
  - Request patterns
  - Stress testing
- [ ] Performance benchmarking
  - Query latency by agent
  - Retrieval time
  - Synthesis time
  - End-to-end response time
- [ ] Optimization based on results
  - Database tuning
  - Connection pooling
  - Caching strategies
  - Parallel execution
- [ ] Create performance report

**Dependencies**: 4.8
**Deliverable**: Performance-tuned system

#### 4.10 Final Integration and Release (Week 16, Day 7)
- [ ] Final end-to-end testing
- [ ] Security audit
- [ ] Code review
- [ ] Version tagging (v1.0.0)
- [ ] Release notes
- [ ] Demo video
- [ ] Launch announcement

**Dependencies**: 4.9
**Deliverable**: Production-ready v1.0.0

---

## Success Criteria

### Phase 1
- ✅ Can index PDF documents from corpus folder
- ✅ Hybrid retrieval returns relevant results
- ✅ LeanHybridAgent produces answers with citations
- ✅ Streaming chat works in browser
- ✅ Logs are clean and informative

### Phase 2
- ✅ RAPTOR summaries improve topical recall
- ✅ Multiple agents available via router
- ✅ Memory persists across sessions
- ✅ Router selects appropriate agents
- ✅ Evidence panel shows citations clearly

### Phase 3
- ✅ GraphRAG captures relationships
- ✅ SelfRAGAgent improves low-confidence answers
- ✅ Validator identifies factual errors
- ✅ Trace viewer shows workflow execution
- ✅ Storage browser allows data exploration

### Phase 4
- ✅ All 13 agents implemented and tested
- ✅ Test suite passes (>80% coverage)
- ✅ System handles 100+ concurrent users
- ✅ Sub-3s response time for simple queries
- ✅ Complete documentation available
- ✅ Production deployment successful

---

## Risk Management

| Risk | Mitigation |
|------|------------|
| LLM API rate limits | Implement retry logic, backoff, caching |
| Database performance | Index optimization, connection pooling, query tuning |
| Vector search latency | HNSW indexes, dimension reduction, caching |
| Complex agent failures | Graceful degradation, fallback agents, error handling |
| Memory bloat | Periodic cleanup, conversation summarization |
| Deployment complexity | Docker containers, clear documentation, automation |

---

## Dependencies Map

```
Phase 1 (Foundation)
    ↓
Phase 2 (Core Agents)
    ├── RAPTOR
    ├── Memory
    ├── Router
    └── Additional Agents
    ↓
Phase 3 (Advanced Features)
    ├── GraphRAG
    ├── SelfRAG
    ├── Validator
    └── Advanced UI
    ↓
Phase 4 (Production)
    ├── Complete Agents
    ├── Testing
    ├── Observability
    └── Deployment
```

---

## Timeline Summary

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| 1 | 4 weeks | Working chat system with LeanHybridAgent |
| 2 | 4 weeks | Multi-agent system with memory and routing |
| 3 | 4 weeks | GraphRAG and advanced features |
| 4 | 4 weeks | Production-ready complete system |
| **Total** | **16 weeks** | **AI-SME System v1.0.0** |

---

## Next Steps

1. ✅ Architecture document created
2. ✅ Roadmap document created
3. → **Start implementation**: Create project structure
4. → Begin Phase 1 tasks
