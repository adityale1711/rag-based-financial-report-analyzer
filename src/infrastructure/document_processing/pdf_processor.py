import hashlib
import pdfplumber
from typing import Any
from pathlib import Path
from ...domain.entities import DocumentChunk
from ...domain.repositories import IDocumentProcessor, DocumentProcessingError


class PDFProcessor(IDocumentProcessor):
    """PDF implementation of the document processor interface.

    This class handles loading and processing PDF documents using pdfplumber
    to extract text and create chunks for RAG implementation.
    """

    def ___init__(self):
        """Initialize the PDF processor."""
        pass

    def _generate_chunk_id(
        self,
        document_name: str,
        page_number: int,
        chunk_index: int
    ) -> str:
        """Generate a unique chunk ID.

        Args:
            document_name: Name of the document.
            page_number: Page number.
            chunk_index: Index of the chunk.

        Returns:
            Unique chunk ID.
        """

        # Create a deterministic ID based on document and position
        unique_string = f"{document_name}_{page_number}_{chunk_index}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:16]

    def _process_single_document(
        self,
        document_path: str
    ) -> list[DocumentChunk]:
        """Process a single PDF document.

        Args:
            document_path: Path to the PDF document.

        Returns:
            List of DocumentChunk objects from the document.
        """
        doc_name = Path(document_path).name
        chunks = []

        try:
            with pdfplumber.open(document_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            # Split the page text into chunks
                            text_chunks = self.chunk_text(text)

                            for chunk_idx, chunk_text in enumerate(text_chunks):
                                chunk_id = self._generate_chunk_id(
                                    doc_name, page_num, chunk_idx
                                )

                                chunk = DocumentChunk(
                                    chunk_id=chunk_id,
                                    document_name=doc_name,
                                    content=chunk_text.strip(),
                                    page_number=page_num,
                                    chunk_index=chunk_idx,
                                    metadata={
                                        "page_num": page_num,
                                        "total_chunks_on_page": len(text_chunks)
                                    }
                                )
                                chunks.append(chunk)
                    except Exception as e:
                        # Log error but continue with other pages
                        print(f"Warning: Failed to process page {page_num}: {str(e)}")
                        continue
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to open PDF {document_path}: {str(e)}"
            ) from e
        
        return chunks

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> list[str]:
        """Split text into chunks for embedding.

        Args:
            text: The text to chunk.
            chunk_size: Maximum size of each chunk.
            overlap: Overlap between chunks.

        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []
        
        # Clean text
        text = text.strip()

        # Split by paragraphs first to maintain context
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(paragraph) + 2 > chunk_size:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Start new chunk with overlap
                if overlap > 0 and current_chunk:
                    # Get last part of current chunk for overlap
                    words = current_chunk.split()
                    overlap_words = []
                    total_length = 0

                    # Work backwards to find overlap text
                    for word in reversed(words):
                        if total_length + len(word) + 1 <= overlap:
                            overlap_words.insert(0, word)
                            total_length += len(word) + 1
                        else:
                            break

                    current_chunk = " ".join(overlap_words) + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

    def process_document(
        self,
        document_paths: list[str]
    ) -> list[DocumentChunk]:
        """Process multiple PDF documents and extract text chunks.

        Args:
            document_paths: List of paths to PDF documents.

        Returns:
            List of DocumentChunk objects containing extracted text.

        Raises:
            DocumentProcessingError: If document processing fails.
        """
        all_chunks = []

        for doc_path in document_paths:
            try:
                chunks = self._process_single_document(doc_path)
                all_chunks.append(chunks)
            except Exception as e:
                raise DocumentProcessingError(
                    f"Failed to process document {doc_path}: {str(e)}"
                )
            
    def extract_metadata(
        self,
        document_path: str
    ) -> dict[str, Any]:
        """Extract metadata from a PDF document.

        Args:
            document_path: Path to the PDF document.

        Returns:
            Dictionary containing document metadata.
        """

        metadata = {
            "document_name": Path(document_path).name,
            "document_path": document_path,
            "file_size": Path(document_path).stat().st_size if Path(document_path).exists() else 0
        }

        try:
            with pdfplumber.open(document_path) as pdf:
                metadata.update({
                    "page_count": len(pdf.pages),
                    "has_text": any(
                        page.extract_text() and page.extract_text().strip()
                        for page in pdf.pages
                    )
                })

                # Try to extract PDF metadata if available
                if hasattr(pdf, 'metadata') and pdf.metadata:
                    pdf_metadata = pdf.metadata
                    metadata.update({
                        "title": pdf_metadata.get('Title', ''),
                        "author": pdf_metadata.get('Author', ''),
                        "creator": pdf_metadata.get('Creator', ''),
                        "producer": pdf_metadata.get('Producer', ''),
                        "creation_date": str(pdf_metadata.get('CreationDate', '')),
                        "modification_date": str(pdf_metadata.get('ModDate', ''))
                    })
        except Exception as e:
            metadata["extraction_error"] = str(e)

        return metadata
