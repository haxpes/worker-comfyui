# AI-SME System - Phase 1

> Multi-agent AI consultant with domain-aware reasoning, hybrid retrieval, and persistent memory.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.28-green.svg)](https://github.com/langchain-ai/langgraph)

## 🎯 Overview

The AI-SME (AI-Subject Matter Expert) system is a domain-aware, multi-agent consultant platform designed to provide explainable, traceable, and contextually relevant answers derived from curated documentation and general knowledge.

### Phase 1 Features (Current)

- ✅ **Hybrid Retrieval**: BM25 + Dense vector search with RRF fusion and MMR diversity
- ✅ **LeanHybridAgent**: Fast, general-purpose agent with cross-encoder reranking
- ✅ **Document Ingestion**: PDF parsing with semantic chunking and OpenAI embeddings
- ✅ **PostgreSQL Backend**: Single unified database with pgvector and full-text search
- ✅ **Streaming API**: FastAPI with Server-Sent Events (SSE) for real-time responses
- ✅ **CLI Tools**: Command-line interface for corpus management and indexing

### Coming in Future Phases

- 🔄 **Phase 2**: RAPTOR summaries, memory system, multi-agent routing
- 🔄 **Phase 3**: GraphRAG, self-reflection, validation
- 🔄 **Phase 4**: Complete agent suite, production deployment

## 🏗️ Architecture

```
User Query
    ↓
FastAPI (SSE Streaming)
    ↓
LeanHybridAgent (LangGraph)
    ├─ Retrieve (BM25 + Dense parallel)
    ├─ Fuse (RRF + MMR)
    ├─ Rerank (Cross-encoder)
    └─ Synthesize (GPT-4o-mini + citations)
    ↓
PostgreSQL (pgvector + pg_search)
```

## 📦 Tech Stack

- **Backend**: FastAPI, LangGraph, LangChain
- **Database**: PostgreSQL 16 with pgvector and pg_search
- **LLM**: OpenAI GPT-4o-mini (synthesis), text-embedding-3-small (embeddings)
- **Retrieval**: BM25 (full-text), Dense (vector), Cross-encoder (reranking)
- **Document Processing**: LlamaIndex (PDF parsing), Semantic chunking

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- OpenAI API key

### Installation

1. **Clone the repository**:
```bash
cd /home/user/worker-comfyui
```

2. **Set up environment**:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

3. **Start PostgreSQL**:
```bash
docker-compose up -d postgres
```

4. **Install Python dependencies**:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

5. **Initialize database**:
```bash
python -m app.db.connection  # Or use CLI: python ../cli.py check
```

### Index Your First Corpus

1. **Prepare your documents**:
```bash
mkdir -p storage/corpus
# Add your PDF files to storage/corpus/
```

2. **Index the corpus**:
```bash
python cli.py index storage/corpus --name "my-corpus"
```

3. **Verify indexing**:
```bash
python cli.py list-corpora
python cli.py info "my-corpus"
```

### Start the API Server

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

### Test the System

**Option 1: Using the API docs**
1. Open http://localhost:8000/docs
2. Try the `/api/chat` endpoint with a query
3. Or use `/api/chat/stream` for streaming responses

**Option 2: Using curl**

Non-streaming:
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the main topics in the corpus?",
    "agent": "lean_hybrid"
  }'
```

Streaming (SSE):
```bash
curl -X POST "http://localhost:8000/api/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the main topics in the corpus?",
    "agent": "lean_hybrid"
  }'
```

## 📁 Project Structure

```
.
├── ARCHITECTURE.md           # Complete system architecture
├── ROADMAP.md               # 16-week implementation plan
├── .env.example             # Environment configuration template
├── docker-compose.yml       # PostgreSQL setup
├── cli.py                   # CLI tool for management
│
├── backend/
│   ├── requirements.txt     # Python dependencies
│   ├── app/
│   │   ├── main.py         # FastAPI application
│   │   ├── config.py       # Configuration management
│   │   │
│   │   ├── core/
│   │   │   ├── state.py    # LangGraph state definitions
│   │   │   └── memory.py   # Memory manager (Phase 2)
│   │   │
│   │   ├── db/
│   │   │   ├── init_db.sql # Database schema
│   │   │   ├── connection.py
│   │   │   └── models.py   # SQLAlchemy ORM
│   │   │
│   │   ├── ingestion/
│   │   │   ├── pdf_parser.py    # PDF extraction
│   │   │   ├── chunker.py       # Semantic chunking
│   │   │   ├── embedder.py      # OpenAI embeddings
│   │   │   └── indexer.py       # Main pipeline
│   │   │
│   │   ├── retrieval/
│   │   │   ├── base.py          # Base retriever
│   │   │   ├── bm25.py          # BM25 retriever
│   │   │   ├── dense.py         # Dense retriever
│   │   │   ├── hybrid.py        # Hybrid retriever
│   │   │   ├── fusion.py        # RRF + MMR
│   │   │   └── reranker.py      # Cross-encoder
│   │   │
│   │   ├── agents/
│   │   │   ├── base.py
│   │   │   └── core/
│   │   │       └── lean_hybrid.py  # LeanHybridAgent
│   │   │
│   │   ├── api/
│   │   │   ├── chat.py          # Chat endpoints
│   │   │   └── agents.py        # Agent management
│   │   │
│   │   └── utils/
│   │       ├── logger.py        # Structured logging
│   │       └── prompts.py       # Prompt templates
│   │
│   └── tests/                   # Tests (Phase 1: manual)
│
├── storage/
│   └── corpus/                  # Your PDF documents
│
└── logs/
    └── app.log                  # Application logs
```

## 🔧 Configuration

Key settings in `.env`:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_sme

# Retrieval Parameters
BM25_K=20          # BM25 top-k results
DENSE_K=20         # Dense retrieval top-k
FUSION_K=10        # After RRF fusion
MMR_K=5            # After MMR diversity
MMR_LAMBDA=0.7     # MMR diversity (0=max diversity, 1=max relevance)
CITATIONS_COUNT=3  # Number of citations in answer

# Models
DEFAULT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
LOG_FORMAT=json    # or 'text'
```

## 📖 CLI Commands

```bash
# Index a corpus
python cli.py index <path> [--name NAME] [--force]

# List all corpora
python cli.py list-corpora

# Get corpus info
python cli.py info <corpus-name>

# System health check
python cli.py check
```

## 🔌 API Endpoints

### Chat Endpoints

- `POST /api/chat` - Non-streaming chat
- `POST /api/chat/stream` - Streaming chat (SSE)
- `GET /api/threads/{thread_id}` - Get conversation (Phase 2)

### Agent Endpoints

- `GET /api/agents` - List all agents
- `GET /api/agents/{agent_name}` - Get agent info

### System Endpoints

- `GET /` - API information
- `GET /health` - Health check

## 📊 Database Schema

**Phase 1 Tables:**

- `corpora` - Corpus metadata
- `documents` - PDF files
- `chunks` - Semantic chunks with embeddings

**Indexes:**

- HNSW index on `chunks.embedding` (vector similarity)
- GIN index on `chunks.content_tsv` (full-text search)

## 🎛️ How It Works

### Document Ingestion Pipeline

1. **PDF Parsing**: Extracts text with structure (headlines, paragraphs)
2. **Semantic Chunking**: Splits into coherent chunks using LlamaIndex
3. **Embedding Generation**: Creates vectors using OpenAI embeddings
4. **Storage**: Saves to PostgreSQL with full-text and vector indexes

### Retrieval Pipeline (LeanHybridAgent)

1. **Parallel Retrieval**:
   - BM25: Keyword-based full-text search
   - Dense: Semantic vector similarity

2. **Fusion & Diversity**:
   - RRF (Reciprocal Rank Fusion): Combines rankings
   - MMR (Maximum Marginal Relevance): Adds diversity

3. **Reranking**:
   - Cross-encoder: Precise query-document matching

4. **Synthesis**:
   - LLM generates answer with citations
   - Extracts citation references

## 🧪 Testing

**Phase 1**: Manual testing only

1. Test indexing:
```bash
python cli.py check
python cli.py index storage/corpus
```

2. Test API:
```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test query", "agent": "lean_hybrid"}'
```

**Phase 2+**: Automated test suite

## 📝 Logging

Logs are written to both console and file (`logs/app.log`).

**Log formats**:
- JSON (default): Structured, machine-readable
- Text: Human-readable with colors

**Log levels**: DEBUG, INFO, WARNING, ERROR

## 🐛 Troubleshooting

### Database Connection Failed

```bash
# Check PostgreSQL is running
docker-compose ps

# Check logs
docker-compose logs postgres

# Restart database
docker-compose restart postgres
```

### Missing PostgreSQL Extensions

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U postgres -d ai_sme

# Check extensions
SELECT * FROM pg_extension;

# Install extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

### Indexing Fails

```bash
# Check corpus path exists
ls storage/corpus

# Check PDF files
ls storage/corpus/*.pdf

# Check OpenAI API key
echo $OPENAI_API_KEY  # or check .env file

# Run with debug logging
LOG_LEVEL=DEBUG python cli.py index storage/corpus
```

## 🗺️ Roadmap

See [ROADMAP.md](ROADMAP.md) for the complete 16-week plan.

**Phase 1** (Weeks 1-4): ✅ COMPLETE
- Hybrid retrieval system
- LeanHybridAgent
- Document ingestion
- FastAPI backend with streaming

**Phase 2** (Weeks 5-8): 🔄 NEXT
- RAPTOR hierarchical summaries
- Memory system (PostgresSaver/Store)
- Multi-agent routing
- Additional agents (RAPTOR, PlanThenRead, EvidenceFirst)

**Phase 3** (Weeks 9-12):
- GraphRAG entity extraction
- Knowledge graphs
- Self-reflection (SelfRAG)
- Validation system

**Phase 4** (Weeks 13-16):
- Complete agent suite
- Testing infrastructure
- Observability (OpenTelemetry, LangSmith)
- Production deployment

## 📚 Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Complete system architecture
- [ROADMAP.md](ROADMAP.md) - Implementation plan
- [API Docs](http://localhost:8000/docs) - Interactive API documentation

## 🤝 Contributing

This is currently a development project. Contributions will be accepted after Phase 1 completion.

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **LangChain/LangGraph**: Orchestration framework
- **OpenAI**: LLM and embeddings
- **pgvector**: Vector similarity in PostgreSQL
- **FastAPI**: Modern Python web framework

---

**Status**: Phase 1 Complete ✅
**Next**: Phase 2 - RAPTOR and Memory Systems 🔄

For questions or issues, see the documentation or check logs in `logs/app.log`.
