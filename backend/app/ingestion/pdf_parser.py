"""
PDF document parser with structure extraction.
Uses LlamaIndex for parsing PDFs with headline and paragraph extraction.
"""
from pathlib import Path
from typing import List, Dict, Any
from llama_index.core import Document as LlamaDocument
from llama_index.readers.file import PDFReader
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PDFParser:
    """
    PDF parser that extracts structured content (headlines and paragraphs).
    """

    def __init__(self):
        """Initialize PDF parser."""
        self.reader = PDFReader()

    def parse_file(self, file_path: str | Path) -> List[Dict[str, Any]]:
        """
        Parse a PDF file and extract structured content.

        Args:
            file_path: Path to PDF file

        Returns:
            List of content blocks with structure metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If parsing fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error("PDF file not found", path=str(file_path))
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logger.info("Parsing PDF file", path=str(file_path))

        try:
            # Load document using LlamaIndex PDFReader
            documents = self.reader.load_data(file=file_path)

            if not documents:
                logger.warning("No content extracted from PDF", path=str(file_path))
                return []

            # Extract structured content
            structured_content = []

            for doc_idx, doc in enumerate(documents):
                # Extract text content
                text = doc.text if hasattr(doc, 'text') else str(doc)
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}

                # Parse into paragraphs (simple split for now)
                # In a more advanced version, we'd use ML models to detect
                # headlines, lists, tables, etc.
                paragraphs = self._split_into_paragraphs(text)

                for para_idx, paragraph in enumerate(paragraphs):
                    if not paragraph.strip():
                        continue

                    # Classify content type (headline vs paragraph)
                    section_type = self._classify_section(paragraph)

                    structured_content.append({
                        "content": paragraph.strip(),
                        "section_type": section_type,
                        "page_number": metadata.get("page_label", doc_idx + 1),
                        "source_file": file_path.name,
                        "metadata": {
                            "doc_index": doc_idx,
                            "para_index": para_idx,
                            **metadata
                        }
                    })

            logger.info(
                "PDF parsing completed",
                path=str(file_path),
                total_blocks=len(structured_content),
                pages=len(documents)
            )

            return structured_content

        except Exception as e:
            logger.error(
                "Failed to parse PDF",
                path=str(file_path),
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.

        Args:
            text: Full text content

        Returns:
            List of paragraphs
        """
        # Split on double newlines (common paragraph separator)
        paragraphs = text.split("\n\n")

        # Also split on single newline if paragraph is very long
        result = []
        for para in paragraphs:
            if len(para) > 1000:  # Long paragraph, try single newline split
                sub_paras = para.split("\n")
                result.extend(sub_paras)
            else:
                result.append(para)

        return [p.strip() for p in result if p.strip()]

    def _classify_section(self, text: str) -> str:
        """
        Classify section type (headline, paragraph, list).

        Args:
            text: Text content

        Returns:
            Section type: "headline", "paragraph", or "list"
        """
        text = text.strip()

        # Headlines are typically short and may be all caps or numbered
        if len(text) < 100:
            # Check for common headline patterns
            if text.isupper():
                return "headline"
            if text[0].isdigit() and "." in text[:10]:  # Numbered section
                return "headline"
            # Check for title case (most words capitalized)
            words = text.split()
            if len(words) <= 10:
                capitalized = sum(1 for w in words if w and w[0].isupper())
                if capitalized / len(words) > 0.5:
                    return "headline"

        # Lists typically have bullet points or numbered items
        list_indicators = ["•", "◦", "▪", "–", "-", "*"]
        if any(text.startswith(indicator) for indicator in list_indicators):
            return "list"

        # Check for numbered lists
        if text[0].isdigit() and (text[1:3] == ". " or text[1:3] == ") "):
            return "list"

        # Default to paragraph
        return "paragraph"

    def parse_directory(
        self,
        directory: str | Path,
        recursive: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parse all PDF files in a directory.

        Args:
            directory: Directory path
            recursive: Whether to search recursively

        Returns:
            Dictionary mapping file paths to parsed content
        """
        directory = Path(directory)

        if not directory.exists():
            logger.error("Directory not found", path=str(directory))
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Find PDF files
        if recursive:
            pdf_files = list(directory.rglob("*.pdf"))
        else:
            pdf_files = list(directory.glob("*.pdf"))

        logger.info(
            "Parsing directory",
            path=str(directory),
            pdf_count=len(pdf_files),
            recursive=recursive
        )

        results = {}
        for pdf_file in pdf_files:
            try:
                content = self.parse_file(pdf_file)
                results[str(pdf_file)] = content
            except Exception as e:
                logger.error(
                    "Failed to parse PDF in directory",
                    path=str(pdf_file),
                    error=str(e)
                )
                # Continue with other files
                continue

        logger.info(
            "Directory parsing completed",
            path=str(directory),
            successful_files=len(results),
            total_files=len(pdf_files)
        )

        return results


def parse_pdf(file_path: str | Path) -> List[Dict[str, Any]]:
    """
    Convenience function to parse a single PDF file.

    Args:
        file_path: Path to PDF file

    Returns:
        List of structured content blocks
    """
    parser = PDFParser()
    return parser.parse_file(file_path)


def parse_pdf_directory(
    directory: str | Path,
    recursive: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convenience function to parse all PDFs in a directory.

    Args:
        directory: Directory path
        recursive: Whether to search recursively

    Returns:
        Dictionary mapping file paths to parsed content
    """
    parser = PDFParser()
    return parser.parse_directory(directory, recursive)
