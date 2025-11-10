-- Phase 3: GraphRAG Tables
-- Entities, relationships, communities for knowledge graph functionality

-- Entities table: Named entities extracted from chunks
CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_id UUID NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL, -- PERSON, ORG, STANDARD, REGULATION, ARTIFACT, etc.
    description TEXT,
    embedding vector(1536),
    confidence FLOAT DEFAULT 1.0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Entity occurrences: Track where entities appear
CREATE TABLE IF NOT EXISTS entity_occurrences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    position INTEGER, -- Position in chunk where entity appears
    context TEXT, -- Surrounding context
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Relationships table: Edges between entities
CREATE TABLE IF NOT EXISTS relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_id UUID NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
    source_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL, -- CO_MENTION, SEQUENTIAL, HIERARCHICAL, CUSTOM
    weight FLOAT DEFAULT 1.0,
    evidence_chunk_ids UUID[], -- Chunks supporting this relationship
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Prevent duplicate relationships
    UNIQUE(source_entity_id, target_entity_id, relationship_type)
);

-- Communities table: Detected entity clusters
CREATE TABLE IF NOT EXISTS communities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_id UUID NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
    name TEXT,
    summary TEXT, -- LLM-generated summary of community
    embedding vector(1536),
    level INTEGER DEFAULT 0, -- Hierarchical level (like RAPTOR)
    size INTEGER, -- Number of entities
    coherence_score FLOAT, -- Community quality metric
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Community members: Entities belonging to communities
CREATE TABLE IF NOT EXISTS community_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    community_id UUID NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    membership_score FLOAT DEFAULT 1.0, -- Strength of membership
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(community_id, entity_id)
);

-- Indexes for performance

-- Entity indexes
CREATE INDEX IF NOT EXISTS idx_entities_corpus ON entities(corpus_id);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_embedding ON entities USING hnsw (embedding vector_cosine_ops);

-- Entity occurrence indexes
CREATE INDEX IF NOT EXISTS idx_entity_occurrences_entity ON entity_occurrences(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_occurrences_chunk ON entity_occurrences(chunk_id);

-- Relationship indexes
CREATE INDEX IF NOT EXISTS idx_relationships_corpus ON relationships(corpus_id);
CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type);

-- Community indexes
CREATE INDEX IF NOT EXISTS idx_communities_corpus ON communities(corpus_id);
CREATE INDEX IF NOT EXISTS idx_communities_embedding ON communities USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_communities_level ON communities(level);

-- Community member indexes
CREATE INDEX IF NOT EXISTS idx_community_members_community ON community_members(community_id);
CREATE INDEX IF NOT EXISTS idx_community_members_entity ON community_members(entity_id);

-- Update trigger for entities
CREATE OR REPLACE FUNCTION update_entities_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER entities_updated_at
    BEFORE UPDATE ON entities
    FOR EACH ROW
    EXECUTE FUNCTION update_entities_updated_at();

-- Update trigger for communities
CREATE TRIGGER communities_updated_at
    BEFORE UPDATE ON communities
    FOR EACH ROW
    EXECUTE FUNCTION update_communities_updated_at();

CREATE OR REPLACE FUNCTION update_communities_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
