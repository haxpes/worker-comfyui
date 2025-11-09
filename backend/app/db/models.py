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
# Phase 2+ Models (will be added later)
# ============================================================================

# These will be added in Phase 2 and beyond:
# - RAPTORSummary
# - RAPTORMembership
# - Entity
# - EntityOccurrence
# - Relationship
# - Community
# - CommunityMember
# - ExperienceKnowledge
# - UserPreference
# - ConversationSummary
# - QueryLog
# - AgentPerformance
# - ValidatorScore
