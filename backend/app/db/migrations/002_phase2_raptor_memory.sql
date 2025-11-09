-- AI-SME System Database Schema - Phase 2 Extensions
-- Adds RAPTOR summaries and memory tables

-- ============================================================================
-- RAPTOR TABLES (Hierarchical Summaries)
-- ============================================================================

-- RAPTOR summaries (per-corpus hierarchical summaries)
CREATE TABLE IF NOT EXISTS raptor_summaries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    corpus_id UUID REFERENCES corpora(id) ON DELETE CASCADE NOT NULL,
    level INTEGER NOT NULL,  -- 1 (chunk clusters), 2 (summary clusters), 3 (corpus-level)
    content TEXT NOT NULL,
    embedding vector(1536),
    parent_id UUID REFERENCES raptor_summaries(id) ON DELETE SET NULL,
    topic VARCHAR(512),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_raptor_summaries_corpus ON raptor_summaries(corpus_id, level);
CREATE INDEX IF NOT EXISTS idx_raptor_summaries_embedding ON raptor_summaries
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_raptor_summaries_parent ON raptor_summaries(parent_id);

-- RAPTOR membership (summaries → chunks)
CREATE TABLE IF NOT EXISTS raptor_membership (
    summary_id UUID REFERENCES raptor_summaries(id) ON DELETE CASCADE NOT NULL,
    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE NOT NULL,
    weight FLOAT DEFAULT 1.0,
    PRIMARY KEY (summary_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_raptor_membership_summary ON raptor_membership(summary_id);
CREATE INDEX IF NOT EXISTS idx_raptor_membership_chunk ON raptor_membership(chunk_id);

-- ============================================================================
-- MEMORY TABLES (LangGraph Integration)
-- ============================================================================

-- User preferences and patterns
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id VARCHAR(255) PRIMARY KEY,
    preferences JSONB NOT NULL DEFAULT '{}'::jsonb,
    query_patterns JSONB DEFAULT '{}'::jsonb,  -- Learned patterns for router bias
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_preferences_updated ON user_preferences(updated_at);

-- Conversation summaries (for context restoration)
CREATE TABLE IF NOT EXISTS conversation_summaries (
    thread_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) REFERENCES user_preferences(user_id) ON DELETE SET NULL,
    summary TEXT NOT NULL,
    embedding vector(1536),
    message_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversation_summaries_user ON conversation_summaries(user_id);
CREATE INDEX IF NOT EXISTS idx_conversation_summaries_embedding ON conversation_summaries
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_conversation_summaries_updated ON conversation_summaries(updated_at);

-- Trigger for conversation summaries updated_at
DROP TRIGGER IF EXISTS update_conversation_summaries_updated_at ON conversation_summaries;
CREATE TRIGGER update_conversation_summaries_updated_at
    BEFORE UPDATE ON conversation_summaries
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger for user preferences updated_at
DROP TRIGGER IF EXISTS update_user_preferences_updated_at ON user_preferences;
CREATE TRIGGER update_user_preferences_updated_at
    BEFORE UPDATE ON user_preferences
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- ANALYTICS TABLES
-- ============================================================================

-- Query logs
CREATE TABLE IF NOT EXISTS query_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255),
    thread_id VARCHAR(255),
    query TEXT NOT NULL,
    refined_query TEXT,
    selected_agents VARCHAR(100)[],
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_query_logs_user ON query_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_thread ON query_logs(thread_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_created ON query_logs(created_at);

-- Agent performance
CREATE TABLE IF NOT EXISTS agent_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_name VARCHAR(100) NOT NULL,
    query_id UUID REFERENCES query_logs(id) ON DELETE CASCADE,
    execution_time_ms INTEGER,
    doc_count INTEGER,
    confidence FLOAT,
    success BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_performance_agent ON agent_performance(agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_performance_query ON agent_performance(query_id);
CREATE INDEX IF NOT EXISTS idx_agent_performance_created ON agent_performance(created_at);

-- Validator scores
CREATE TABLE IF NOT EXISTS validator_scores (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id UUID REFERENCES query_logs(id) ON DELETE CASCADE,
    factuality FLOAT,
    consistency FLOAT,
    completeness FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_validator_scores_query ON validator_scores(query_id);

-- ============================================================================
-- SCHEMA VERSION UPDATE
-- ============================================================================

INSERT INTO schema_version (version, description)
VALUES (2, 'Phase 2: RAPTOR summaries, memory tables, and analytics')
ON CONFLICT (version) DO NOTHING;

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Phase 2 schema extensions applied successfully!';
    RAISE NOTICE 'New tables: raptor_summaries, raptor_membership, user_preferences, conversation_summaries, query_logs, agent_performance, validator_scores';
END $$;
