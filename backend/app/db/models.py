"""
SQLAlchemy ORM models for the AI-SME system.
"""
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class Corpus(Base):
    """Corpus: Container for related documents."""
    __tablename__ = "corpora"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()")
    )
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()")
    )

    # Relationships
    documents: Mapped[List["Document"]] = relationship(
        "Document",
        back_populates="corpus",
        cascade="all, delete-orphan"
    )
    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk",
        back_populates="corpus",
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Corpus(id={self.id}, name='{self.name}')>"


class Document(Base):
    """Document: File in a corpus."""
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()")
    )
    corpus_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("corpora.id", ondelete="CASCADE"),
        nullable=False
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(Text)
    doc_type: Mapped[Optional[str]] = mapped_column(String(50))
    metadata: Mapped[dict] = mapped_column(
        JSONB,
        server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()")
    )

    # Relationships
    corpus: Mapped["Corpus"] = relationship("Corpus", back_populates="documents")
    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename='{self.filename}')>"


class Chunk(Base):
    """Chunk: Semantic chunk of a document with embeddings."""
    __tablename__ = "chunks"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()")
    )
    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False
    )
    corpus_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("corpora.id", ondelete="CASCADE"),
        nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(1536))
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    section_type: Mapped[Optional[str]] = mapped_column(String(50))
    metadata: Mapped[dict] = mapped_column(
        JSONB,
        server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()")
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
    corpus: Mapped["Corpus"] = relationship("Corpus", back_populates="chunks")

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<Chunk(id={self.id}, index={self.chunk_index}, content='{preview}')>"


# ============================================================================
# Phase 2 Models - RAPTOR
# ============================================================================

class RAPTORSummary(Base):
    """RAPTOR hierarchical summary."""
    __tablename__ = "raptor_summaries"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()")
    )
    corpus_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("corpora.id", ondelete="CASCADE"),
        nullable=False
    )
    level: Mapped[int] = mapped_column(Integer, nullable=False)  # 1, 2, 3
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(1536))
    parent_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("raptor_summaries.id", ondelete="SET NULL")
    )
    topic: Mapped[Optional[str]] = mapped_column(String(512))
    metadata: Mapped[dict] = mapped_column(
        JSONB,
        server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()")
    )

    # Relationships
    corpus: Mapped["Corpus"] = relationship("Corpus")
    parent: Mapped[Optional["RAPTORSummary"]] = relationship(
        "RAPTORSummary",
        remote_side=[id],
        back_populates="children"
    )
    children: Mapped[List["RAPTORSummary"]] = relationship(
        "RAPTORSummary",
        back_populates="parent"
    )

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<RAPTORSummary(id={self.id}, level={self.level}, topic='{self.topic}')>"


class RAPTORMembership(Base):
    """RAPTOR summary to chunk membership."""
    __tablename__ = "raptor_membership"

    summary_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("raptor_summaries.id", ondelete="CASCADE"),
        primary_key=True
    )
    chunk_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        primary_key=True
    )
    weight: Mapped[float] = mapped_column(Float, default=1.0)


# ============================================================================
# Phase 2 Models - Memory
# ============================================================================

class UserPreference(Base):
    """User preferences and learned query patterns."""
    __tablename__ = "user_preferences"

    user_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    preferences: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb")
    )
    query_patterns: Mapped[dict] = mapped_column(
        JSONB,
        server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()")
    )

    def __repr__(self) -> str:
        return f"<UserPreference(user_id='{self.user_id}')>"


class ConversationSummary(Base):
    """Conversation summaries for context restoration."""
    __tablename__ = "conversation_summaries"

    thread_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    user_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        ForeignKey("user_preferences.user_id", ondelete="SET NULL")
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(1536))
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()")
    )

    def __repr__(self) -> str:
        return f"<ConversationSummary(thread_id='{self.thread_id}', messages={self.message_count})>"


# ============================================================================
# Phase 2 Models - Analytics
# ============================================================================

class QueryLog(Base):
    """Query execution logs."""
    __tablename__ = "query_logs"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()")
    )
    user_id: Mapped[Optional[str]] = mapped_column(String(255))
    thread_id: Mapped[Optional[str]] = mapped_column(String(255))
    query: Mapped[str] = mapped_column(Text, nullable=False)
    refined_query: Mapped[Optional[str]] = mapped_column(Text)
    selected_agents: Mapped[Optional[List[str]]] = mapped_column(JSONB)
    execution_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()")
    )

    def __repr__(self) -> str:
        return f"<QueryLog(id={self.id}, query='{self.query[:50]}...')>"


class AgentPerformance(Base):
    """Agent performance metrics."""
    __tablename__ = "agent_performance"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()")
    )
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False)
    query_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("query_logs.id", ondelete="CASCADE")
    )
    execution_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    doc_count: Mapped[Optional[int]] = mapped_column(Integer)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    success: Mapped[Optional[bool]] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()")
    )

    def __repr__(self) -> str:
        return f"<AgentPerformance(agent='{self.agent_name}', success={self.success})>"


class ValidatorScore(Base):
    """Validation scores for answers."""
    __tablename__ = "validator_scores"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()")
    )
    query_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("query_logs.id", ondelete="CASCADE")
    )
    factuality: Mapped[Optional[float]] = mapped_column(Float)
    consistency: Mapped[Optional[float]] = mapped_column(Float)
    completeness: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()")
    )

    def __repr__(self) -> str:
        return f"<ValidatorScore(factuality={self.factuality}, consistency={self.consistency})>"


# ============================================================================
# Phase 3+ Models (will be added later)
# ============================================================================

# Phase 3: GraphRAG
# - Entity
# - EntityOccurrence
# - Relationship
# - Community
# - CommunityMember
# - ExperienceKnowledge
