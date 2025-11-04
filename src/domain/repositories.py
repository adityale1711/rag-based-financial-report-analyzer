from abc import ABC, abstractmethod
from typing import Any, Optional
from .entities import (
    DocumentChunk,
    Question,
    RetrievalResult,
    RAGAnswer,
    Visualization,
    VisualizationType
)


class ILLMProvider(ABC):
    """Interface for Large Language Model providers.

    This interface defines the contract for LLM services that can process
    natural language questions and generate answers with analysis code.
    """

    @abstractmethod
    async def generate_rag_answer(
        self,
        rag_prompt: str
    ) -> RAGAnswer:
        """Generate an answer using RAG context.

        Args:
            rag_prompt: The complete RAG prompt including context and question.

        Returns:
            A RAG answer based on retrieved documents.

        Raises:
            LLMProviderError: If the LLM provider fails to generate an answer.
        """
        pass






class IChartGenerator(ABC):
    """Interface for generating data .

    This interface defines the contract for creating charts and graphs
    based on analysis results.
    """

    @abstractmethod
    def generate_chart(
        self,
        chart_type: VisualizationType,
        data: Any,
        title: str,
        config: Optional[dict[str, Any]] = None
    ) -> Visualization:
        """Generate a visualization based on the data.

        Args:
            chart_type: Type of chart to generate.
            data: Data to visualize.
            title: Chart title.
            config: Additional chart configuration.

        Returns:
            Visualization object with the generated chart.

        Raises:
            ChartGenerationError: If chart generation fails.
        """
        pass




class IDocumentProcessor(ABC):
    """Interface for document processing operations.

    This interface defines the contract for processing PDF documents
    and extracting text chunks for RAG implementation.
    """

    @abstractmethod
    def process_documents(
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
        pass

    @abstractmethod
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
        pass


class IVectorStore(ABC):
    """Interface for vector storage and retrieval operations.

    This interface defines the contract for storing document embeddings
    and retrieving relevant chunks based on queries.
    """

    @abstractmethod
    def add_documents(
        self,
        chunks: list[DocumentChunk]
    ) -> None:
        """Add document chunks to the vector store.

        Args:
            chunks: List of document chunks to add.

        Raises:
            VectorStoreError: If adding documents fails.
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        n_results: int = 5
    ) -> RetrievalResult:
        """Search for relevant document chunks.

        Args:
            query: The search query.
            n_results: Number of results to return.

        Returns:
            RetrievalResult with relevant document chunks.

        Raises:
            VectorStoreError: If search fails.
        """
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """Check if the vector store is empty.

        Returns:
            True if the vector store has no documents, False otherwise.
        """
        pass


class IRAGService(ABC):
    """Interface for RAG (Retrieval-Augmented Generation) operations.

    This interface defines the contract for the complete RAG pipeline.
    """

    @abstractmethod
    def initialize(
        self,
        document_paths: list[str]
    ) -> None:
        """Initialize the RAG service by processing documents.

        Args:
            document_paths: List of paths to PDF documents.

        Raises:
            RAGInitializationError: If initialization fails.
        """
        pass

    @abstractmethod
    async def process_question(
        self,
        question: Question
    ) -> RAGAnswer:
        """Process a question using RAG.

        Args:
            question: The user's question.

        Returns:
            RAG answer based on retrieved documents.

        Raises:
            RAGServiceError: If processing fails.
        """
        pass


# Custom exception classes for better error handling
class LLMProviderError(Exception):
    """Exception raised for errors in the LLM provider."""
    pass




class ChartGenerationError(Exception):
    """Exception raised for errors in chart generation."""
    pass


class DocumentProcessingError(Exception):
    """Exception raised for errors in document processing."""
    pass

class VectorStoreError(Exception):
    """Exception raised for errors in vector store operations."""
    pass

class RAGServiceError(Exception):
    """Exception raised for errors in the RAG service."""
    pass
