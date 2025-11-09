"""
Main indexing pipeline: parse → chunk → embed → store.
Orchestrates the entire document ingestion process.
"""
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from datetime import datetime
from sqlalchemy import select
from app.config import settings
from app.db.connection import get_db_session
from app.db.models import Corpus, Document, Chunk
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import SemanticChunker
from app.ingestion.embedder import get_embedder
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentIndexer:
    """
    Main indexing pipeline for documents.
    Handles: parsing → chunking → embedding → database storage.
    """

    def __init__(self):
        """Initialize indexer with all components."""
        self.pdf_parser = PDFParser()
        self.chunker = SemanticChunker()
        self.embedder = get_embedder()

    async def index_corpus(
        self,
        corpus_path: str | Path,
        corpus_name: str | None = None,
        corpus_description: str | None = None,
        force_reindex: bool = False
    ) -> Dict[str, Any]:
        """
        Index an entire corpus directory.

        Args:
            corpus_path: Path to corpus directory
            corpus_name: Name for the corpus (defaults to directory name)
            corpus_description: Optional description
            force_reindex: If True, delete and reindex existing corpus

        Returns:
            Dictionary with indexing statistics
        """
        corpus_path = Path(corpus_path)

        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus path not found: {corpus_path}")

        if not corpus_path.is_dir():
            raise ValueError(f"Corpus path must be a directory: {corpus_path}")

        # Use directory name as corpus name if not provided
        if corpus_name is None:
            corpus_name = corpus_path.name

        logger.info(
            "Starting corpus indexing",
            corpus_name=corpus_name,
            corpus_path=str(corpus_path),
            force_reindex=force_reindex
        )

        start_time = datetime.now()

        # Create or get corpus
        corpus_id = await self._ensure_corpus(
            name=corpus_name,
            description=corpus_description,
            file_path=str(corpus_path),
            force_reindex=force_reindex
        )

        # Find all PDF files
        pdf_files = list(corpus_path.glob("*.pdf"))

        if not pdf_files:
            logger.warning(
                "No PDF files found in corpus",
                corpus_path=str(corpus_path)
            )
            return {
                "corpus_id": str(corpus_id),
                "corpus_name": corpus_name,
                "documents_indexed": 0,
                "chunks_created": 0,
                "duration_seconds": 0
            }

        logger.info(
            "Found PDF files",
            count=len(pdf_files),
            corpus_name=corpus_name
        )

        # Index each document
        total_chunks = 0
        for pdf_file in pdf_files:
            try:
                chunks_count = await self.index_document(
                    file_path=pdf_file,
                    corpus_id=corpus_id
                )
                total_chunks += chunks_count

                logger.info(
                    "Document indexed",
                    filename=pdf_file.name,
                    chunks=chunks_count
                )

            except Exception as e:
                logger.error(
                    "Failed to index document",
                    filename=pdf_file.name,
                    error=str(e)
                )
                # Continue with other documents

        duration = (datetime.now() - start_time).total_seconds()

        stats = {
            "corpus_id": str(corpus_id),
            "corpus_name": corpus_name,
            "documents_indexed": len(pdf_files),
            "chunks_created": total_chunks,
            "duration_seconds": round(duration, 2)
        }

        logger.info(
            "Corpus indexing completed",
            **stats
        )

        return stats

    async def index_document(
        self,
        file_path: str | Path,
        corpus_id: UUID
    ) -> int:
        """
        Index a single document: parse → chunk → embed → store.

        Args:
            file_path: Path to document file
            corpus_id: UUID of corpus this document belongs to

        Returns:
            Number of chunks created
        """
        file_path = Path(file_path)

        logger.info(
            "Indexing document",
            filename=file_path.name,
            corpus_id=str(corpus_id)
        )

        # Step 1: Parse PDF
        logger.debug("Parsing PDF...", filename=file_path.name)
        structured_content = self.pdf_parser.parse_file(file_path)

        if not structured_content:
            logger.warning("No content extracted", filename=file_path.name)
            return 0

        # Step 2: Chunk content
        logger.debug(
            "Chunking content...",
            filename=file_path.name,
            blocks=len(structured_content)
        )
        chunks_data = self.chunker.chunk_by_section_type(
            structured_content,
            chunk_headlines=False  # Keep headlines intact
        )

        if not chunks_data:
            logger.warning("No chunks created", filename=file_path.name)
            return 0

        # Step 3: Generate embeddings
        logger.debug(
            "Generating embeddings...",
            filename=file_path.name,
            chunks=len(chunks_data)
        )
        texts = [chunk["content"] for chunk in chunks_data]
        embeddings = await self.embedder.embed_documents(texts, show_progress=False)

        # Step 4: Store in database
        logger.debug(
            "Storing in database...",
            filename=file_path.name,
            chunks=len(chunks_data)
        )

        async with get_db_session() as session:
            # Create document record
            doc = Document(
                corpus_id=corpus_id,
                filename=file_path.name,
                file_path=str(file_path),
                title=file_path.stem,
                doc_type="pdf",
                metadata={
                    "total_chunks": len(chunks_data),
                    "total_blocks": len(structured_content),
                }
            )
            session.add(doc)
            await session.flush()  # Get document ID

            # Create chunk records
            chunk_records = []
            for chunk_data, embedding in zip(chunks_data, embeddings):
                chunk = Chunk(
                    document_id=doc.id,
                    corpus_id=corpus_id,
                    content=chunk_data["content"],
                    embedding=embedding,
                    chunk_index=chunk_data["chunk_index"],
                    section_type=chunk_data.get("section_type", "paragraph"),
                    metadata=chunk_data.get("metadata", {})
                )
                chunk_records.append(chunk)

            session.add_all(chunk_records)
            await session.commit()

        logger.info(
            "Document indexed successfully",
            filename=file_path.name,
            document_id=str(doc.id),
            chunks_created=len(chunk_records)
        )

        return len(chunk_records)

    async def _ensure_corpus(
        self,
        name: str,
        description: Optional[str],
        file_path: str,
        force_reindex: bool
    ) -> UUID:
        """
        Create or get corpus, optionally deleting existing one.

        Args:
            name: Corpus name
            description: Corpus description
            file_path: Corpus file path
            force_reindex: Whether to delete existing corpus

        Returns:
            Corpus UUID
        """
        async with get_db_session() as session:
            # Check if corpus exists
            result = await session.execute(
                select(Corpus).where(Corpus.name == name)
            )
            existing_corpus = result.scalar_one_or_none()

            if existing_corpus:
                if force_reindex:
                    logger.warning(
                        "Deleting existing corpus for reindexing",
                        corpus_name=name,
                        corpus_id=str(existing_corpus.id)
                    )
                    await session.delete(existing_corpus)
                    await session.commit()
                    existing_corpus = None
                else:
                    logger.info(
                        "Using existing corpus",
                        corpus_name=name,
                        corpus_id=str(existing_corpus.id)
                    )
                    return existing_corpus.id

            # Create new corpus
            corpus = Corpus(
                name=name,
                description=description,
                file_path=file_path
            )
            session.add(corpus)
            await session.commit()
            await session.refresh(corpus)

            logger.info(
                "Corpus created",
                corpus_name=name,
                corpus_id=str(corpus.id)
            )

            return corpus.id

    async def get_corpus_stats(self, corpus_id: UUID) -> Dict[str, Any]:
        """
        Get statistics for a corpus.

        Args:
            corpus_id: Corpus UUID

        Returns:
            Statistics dictionary
        """
        async with get_db_session() as session:
            # Get corpus
            result = await session.execute(
                select(Corpus).where(Corpus.id == corpus_id)
            )
            corpus = result.scalar_one_or_none()

            if not corpus:
                raise ValueError(f"Corpus not found: {corpus_id}")

            # Count documents
            doc_result = await session.execute(
                select(Document).where(Document.corpus_id == corpus_id)
            )
            documents = doc_result.scalars().all()

            # Count chunks
            chunk_result = await session.execute(
                select(Chunk).where(Chunk.corpus_id == corpus_id)
            )
            chunks = chunk_result.scalars().all()

            return {
                "corpus_id": str(corpus.id),
                "corpus_name": corpus.name,
                "description": corpus.description,
                "file_path": corpus.file_path,
                "document_count": len(documents),
                "chunk_count": len(chunks),
                "created_at": corpus.created_at.isoformat(),
                "updated_at": corpus.updated_at.isoformat(),
            }


async def index_corpus(
    corpus_path: str | Path,
    corpus_name: str | None = None,
    force_reindex: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to index a corpus.

    Args:
        corpus_path: Path to corpus directory
        corpus_name: Optional corpus name
        force_reindex: Whether to force reindexing

    Returns:
        Indexing statistics
    """
    indexer = DocumentIndexer()
    return await indexer.index_corpus(
        corpus_path=corpus_path,
        corpus_name=corpus_name,
        force_reindex=force_reindex
    )
