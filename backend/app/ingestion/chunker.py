"""
Semantic chunking for documents.
Uses LlamaIndex SemanticSplitterNodeParser for coherence-based chunking.
"""
from typing import List, Dict, Any
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document as LlamaDocument, TextNode
from app.ingestion.embedder import get_embedder
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SemanticChunker:
    """
    Semantic chunker that splits documents based on semantic coherence.
    No max chunk size, no overlaps (as per requirements).
    """

    def __init__(self):
        """Initialize semantic chunker."""
        self.embedder = get_embedder()

        # Create embedding function compatible with LlamaIndex
        self._init_splitter()

    def _init_splitter(self):
        """Initialize the semantic splitter with custom embed function."""
        from llama_index.embeddings.openai import OpenAIEmbedding
        from app.config import settings

        # Use LlamaIndex's OpenAI embedding wrapper for compatibility
        embed_model = OpenAIEmbedding(
            model=settings.embedding_model,
            api_key=settings.openai_api_key
        )

        # Create semantic splitter
        # breakpoint_percentile_threshold: Lower = more chunks, Higher = fewer chunks
        # buffer_size: Number of sentences to group together
        self.splitter = SemanticSplitterNodeParser(
            buffer_size=1,  # Sentences to group
            breakpoint_percentile_threshold=95,  # Coherence threshold
            embed_model=embed_model,
        )

        logger.info("Semantic chunker initialized")

    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk a single text into semantically coherent pieces.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of chunks with metadata
        """
        if not text.strip():
            return []

        metadata = metadata or {}

        logger.debug(
            "Chunking text",
            text_length=len(text),
            metadata_keys=list(metadata.keys())
        )

        try:
            # Create LlamaIndex document
            doc = LlamaDocument(text=text, metadata=metadata)

            # Split into semantic chunks
            nodes = self.splitter.get_nodes_from_documents([doc])

            # Convert nodes to our format
            chunks = []
            for idx, node in enumerate(nodes):
                chunk_data = {
                    "content": node.text,
                    "chunk_index": idx,
                    "metadata": {
                        **metadata,
                        "start_char_idx": node.start_char_idx,
                        "end_char_idx": node.end_char_idx,
                        "chunk_length": len(node.text),
                    }
                }
                chunks.append(chunk_data)

            logger.debug(
                "Text chunked",
                original_length=len(text),
                num_chunks=len(chunks)
            )

            return chunks

        except Exception as e:
            logger.error(
                "Failed to chunk text",
                error=str(e),
                text_length=len(text)
            )
            raise

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents (structured content blocks).

        Args:
            documents: List of document dictionaries with 'content' and optional metadata

        Returns:
            List of all chunks from all documents
        """
        if not documents:
            return []

        logger.info(
            "Chunking documents",
            num_documents=len(documents)
        )

        all_chunks = []

        for doc_idx, doc in enumerate(documents):
            content = doc.get("content", "")
            if not content.strip():
                continue

            # Merge document metadata
            doc_metadata = {
                "source_document_index": doc_idx,
                "section_type": doc.get("section_type", "paragraph"),
                **doc.get("metadata", {})
            }

            # Chunk this document
            chunks = self.chunk_text(content, metadata=doc_metadata)

            # Add global chunk index
            for chunk in chunks:
                chunk["global_chunk_index"] = len(all_chunks)
                all_chunks.append(chunk)

        logger.info(
            "Document chunking completed",
            num_documents=len(documents),
            total_chunks=len(all_chunks),
            avg_chunks_per_doc=round(len(all_chunks) / len(documents), 2) if documents else 0
        )

        return all_chunks

    def chunk_by_section_type(
        self,
        structured_content: List[Dict[str, Any]],
        chunk_headlines: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Chunk structured content, optionally treating headlines differently.

        Args:
            structured_content: List of content blocks with section_type
            chunk_headlines: If False, keep headlines as single chunks

        Returns:
            List of chunks
        """
        all_chunks = []

        for block_idx, block in enumerate(structured_content):
            content = block.get("content", "")
            section_type = block.get("section_type", "paragraph")

            if not content.strip():
                continue

            # Headlines: keep as-is unless chunking is enabled
            if section_type == "headline" and not chunk_headlines:
                chunk = {
                    "content": content,
                    "chunk_index": 0,
                    "global_chunk_index": len(all_chunks),
                    "section_type": section_type,
                    "metadata": {
                        **block.get("metadata", {}),
                        "source_block_index": block_idx,
                    }
                }
                all_chunks.append(chunk)
            else:
                # Chunk normally
                metadata = {
                    "source_block_index": block_idx,
                    "section_type": section_type,
                    **block.get("metadata", {})
                }

                chunks = self.chunk_text(content, metadata=metadata)

                for chunk in chunks:
                    chunk["global_chunk_index"] = len(all_chunks)
                    all_chunks.append(chunk)

        logger.info(
            "Section-based chunking completed",
            total_blocks=len(structured_content),
            total_chunks=len(all_chunks)
        )

        return all_chunks


def chunk_text(text: str, metadata: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """
    Convenience function to chunk a single text.

    Args:
        text: Text to chunk
        metadata: Optional metadata

    Returns:
        List of chunks
    """
    chunker = SemanticChunker()
    return chunker.chunk_text(text, metadata)


def chunk_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience function to chunk multiple documents.

    Args:
        documents: List of documents

    Returns:
        List of all chunks
    """
    chunker = SemanticChunker()
    return chunker.chunk_documents(documents)
