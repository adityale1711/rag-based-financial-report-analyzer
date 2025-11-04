from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass


class VisualizationType(Enum):
    """Enumeration of supported visualization types."""
    
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"


class ProcessingStatus(Enum):
    """Enumeration of processing statuses."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class Question:
    """Represents a user question about the dataset.

    Attributes:
        text: The natural language question text.
        language: The language of the question (e.g., 'en', 'id').
    """

    text: str
    language: str = "en"




@dataclass(frozen=True)
class Visualization:
    """Represents a data visualization.

    Attributes:
        chart_type: The type of chart (bar, line, pie, etc.).
        chart_object: The plotly chart object.
        title: The title of the chart.
        description: Description of what the visualization shows.
        config: Additional configuration for the chart.
    """

    chart_type: VisualizationType
    chart_object: Any
    title: str
    description: str
    config: Optional[dict[str, Any]] = None






@dataclass(frozen=True)
class ProcessingError:
    """Represents an error that occurred during processing.

    Attributes:
        error_type: Type of error (e.g., 'api_error', 'code_execution').
        message: Human-readable error message.
        details: Additional technical details.
        retry_suggested: Whether retrying might help.
    """

    error_type: str
    message: str
    details: Optional[str] = None
    retry_suggested: bool = False


@dataclass(frozen=True)
class FinancialDataPoint:
    """Represents a single financial data point extracted from a document.

    Attributes:
        metric_type: Type of financial metric (e.g., "total_assets", "net_profit").
        value: Numerical value of the metric.
        period: Time period for the metric (e.g., "Aug 2024", "Q3 2024").
        currency: Currency code (default: "IDR").
        confidence: Confidence score for the extraction (0.0 to 1.0).
        raw_text: Original text from which the data was extracted.
    """

    metric_type: str
    value: float
    period: str
    currency: str = "IDR"
    confidence: float = 1.0
    raw_text: str = ""


@dataclass(frozen=True)
class StructuredFinancialData:
    """Structured financial data extracted from document chunks.

    Attributes:
        data_points: List of financial data points extracted from the chunk.
        document_name: Name of the source document.
        extraction_method: Method used for extraction (e.g., "regex", "ml").
        confidence_score: Overall confidence score for the extracted data.
        extraction_timestamp: When the data was extracted.
    """

    data_points: list[FinancialDataPoint]
    document_name: str
    extraction_method: str = "regex"
    confidence_score: float = 1.0
    extraction_timestamp: Optional[str] = None


@dataclass(frozen=True)
class DocumentChunk:
    """Represents a chunk of text extracted from a document.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        document_name: Name of the source document.
        content: The text content of the chunk.
        page_number: Page number where the chunk was found.
        chunk_index: Index of the chunk within the document.
        financial_data: Structured financial data extracted from this chunk.
        metadata: Additional metadata about the chunk.
    """

    chunk_id: str
    document_name: str
    content: str
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    financial_data: Optional[StructuredFinancialData] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class RAGAnswer:
    """Represents an answer generated using RAG.

    Attributes:
        text: The textual answer based on retrieved documents.
        confidence_score: A score from 0 to 1 indicating confidence.
        sources: List of source document chunks used.
        explanation: Additional explanation of how the answer was derived.
    """

    text: str
    confidence_score: float
    sources: list[DocumentChunk]
    explanation: Optional[str] = None


@dataclass(frozen=True)
class RetrievalResult:
    """Represents the result of retrieving relevant document chunks.

    Attributes:
        chunks: List of retrieved document chunks.
        query: The original query used for retrieval.
        retrieval_score: Average relevance score of retrieved chunks.
        total_retrieved: Total number of chunks retrieved.
    """

    chunks: list[DocumentChunk]
    query: str
    retrieval_score: float
    total_retrieved: int


@dataclass(frozen=True)
class AnalysisResult:
    """Represents the complete result of analyzing a question using RAG.

    This is the main entity that combines all components of the RAG analysis.

    Attributes:
        question: The original question.
        rag_answer: The RAG-generated answer with sources.
        visualization: The generated visualization.
        execution_time: Time taken to process the question (in seconds).
    """

    question: Question
    rag_answer: RAGAnswer
    visualization: Optional[Visualization]
    execution_time: float
