-- AI-SME System Database Schema - Phase 1
-- Minimal schema for foundation: documents, chunks, and basic indexes

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For full-text search

-- ============================================================================
-- CORPUS AND DOCUMENTS
-- ============================================================================

-- Corpus metadata (container for related documents)
CREATE TABLE IF NOT EXISTS corpora (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    file_path TEXT NOT NULL,  -- e.g., /storage/medical-standards/
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_corpora_name ON corpora(name);

-- Documents (references to files in corpus folders)
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    corpus_id UUID REFERENCES corpora(id) ON DELETE CASCADE,
    filename VARCHAR(512) NOT NULL,
    file_path TEXT NOT NULL,
    title TEXT,
    doc_type VARCHAR(50),  -- pdf, markdown, etc.
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(corpus_id, filename)
);

CREATE INDEX IF NOT EXISTS idx_documents_corpus ON documents(corpus_id);
CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING gin(metadata);

-- ============================================================================
-- CHUNKS (Global searchable)
-- ============================================================================

-- Chunks with vector embeddings and full-text search
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE NOT NULL,
    corpus_id UUID REFERENCES corpora(id) ON DELETE CASCADE NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),  -- text-embedding-3-small dimension
    chunk_index INTEGER NOT NULL,
    section_type VARCHAR(50),  -- headline, paragraph, list, etc.
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    -- Full-text search vector (auto-generated)
    content_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', content)
    ) STORED
);

-- Indexes for chunks
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_corpus ON chunks(corpus_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);  -- HNSW parameters for pgvector
CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv ON chunks USING gin(content_tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON chunks USING gin(metadata);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for corpora updated_at
DROP TRIGGER IF EXISTS update_corpora_updated_at ON corpora;
CREATE TRIGGER update_corpora_updated_at
    BEFORE UPDATE ON corpora
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- VIEWS FOR CONVENIENCE
-- ============================================================================

-- View combining chunks with document and corpus info
CREATE OR REPLACE VIEW chunks_with_metadata AS
SELECT
    c.id as chunk_id,
    c.content,
    c.embedding,
    c.chunk_index,
    c.section_type,
    c.metadata as chunk_metadata,
    d.id as document_id,
    d.filename,
    d.title as document_title,
    d.doc_type,
    d.metadata as document_metadata,
    co.id as corpus_id,
    co.name as corpus_name,
    co.description as corpus_description,
    c.created_at
FROM chunks c
JOIN documents d ON c.document_id = d.id
JOIN corpora co ON c.corpus_id = co.id;

-- ============================================================================
-- SAMPLE DATA (for testing)
-- ============================================================================

-- Insert default corpus
INSERT INTO corpora (name, description, file_path)
VALUES (
    'general',
    'General knowledge corpus',
    '/home/user/worker-comfyui/storage/corpus'
) ON CONFLICT (name) DO NOTHING;

-- ============================================================================
-- GRANTS AND PERMISSIONS
-- ============================================================================

-- Grant permissions (adjust user as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ai_sme_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ai_sme_user;

-- ============================================================================
-- SCHEMA VERSION
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    description TEXT
);

INSERT INTO schema_version (version, description)
VALUES (1, 'Phase 1: Basic schema with corpora, documents, and chunks')
ON CONFLICT (version) DO NOTHING;

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE 'AI-SME Database Schema Phase 1 initialized successfully!';
    RAISE NOTICE 'Extensions: uuid-ossp, vector, pg_trgm';
    RAISE NOTICE 'Tables: corpora, documents, chunks';
    RAISE NOTICE 'Ready for document ingestion.';
END $$;
