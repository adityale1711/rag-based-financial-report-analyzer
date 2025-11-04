import re
from typing import Any
from ... import logger
from ...domain.entities import (
    DocumentChunk,
    Question,
    RAGAnswer
)
from ...domain.repositories import (
    IDocumentProcessor,
    ILLMProvider,
    IRAGService,
    IVectorStore,
    RAGServiceError
)


class RAGService(IRAGService):
    """Implementation of the RAG (Retrieval-Augmented Generation) service.

    This class orchestrates the complete RAG pipeline including document processing,
    vector storage, retrieval, and LLM-based answer generation.
    """

    def __init__(
        self,
        document_processor: IDocumentProcessor,
        vector_store: IVectorStore,
        llm_provider: ILLMProvider
    ):
        """Initialize the RAG service.

        Args:
            document_processor: Service for processing PDF documents.
            vector_store: Service for storing and retrieving document embeddings.
            llm_provider: Service for LLM interactions.
        """
        self.document_processor = document_processor
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self._is_initialized = False

    def _build_context_text(
        self,
        chunks: list[DocumentChunk]
    ) -> str:
        """Build context text from retrieved chunks.

        Args:
            chunks: List of retrieved document chunks.

        Returns:
            Formatted context text.
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source_info = f"(Source: {chunk.document_name}"
            if chunk.page_number:
                source_info += f", Page: {chunk.page_number}"

            context_parts.append(
                f"[Excerpt {i}] {source_info})\n{chunk.content}\n"
            )

        return "\n".join(context_parts)
    
    def _aggregate_financial_data_from_chunks(
        self,
        chunks: list[DocumentChunk],
        metric_type: str
    ) -> dict:
        """Aggregate financial data from document chunks for visualization.

        Args:
            chunks: List of document chunks with pre-extracted financial data.
            metric_type: The metric type to aggregate (e.g., "total_assets").

        Returns:
            Dictionary with aggregated financial data ready for visualization.
        """
        from src.infrastructure.document_processing.financial_extractor import FinancialDataExtractor

        # Collect all financial data from chunks
        financial_data_list = []
        for chunk in chunks:
            if chunk.financial_data and chunk.financial_data.data_points:
                financial_data_list.append(chunk.financial_data)

        if financial_data_list:
            return FinancialDataExtractor.aggregate_financial_data_by_metric(
                financial_data_list, metric_type
            )

        return None
        
    def _build_rag_prompt(
        self,
        question: str,
        context: str,
        financial_data: dict = None
    ) -> str:
        """Build a RAG-specific prompt for the LLM.

        Args:
            question: The user's question.
            context: Retrieved context from documents.
            financial_data: Extracted financial data for visualization.

        Returns:
            Complete RAG prompt.
        """
        try:
            # Load the RAG prompt template from the configured file path
            with open('prompts/application/services/rag_prompt.txt', 'r') as file:
                prompt_template = file.read()

            financial_data = financial_data if financial_data else "No structured financial data available"

            # Format the template with the specific arguments
            prompt = prompt_template.format(
                question=question,
                context=context,
                financial_data=financial_data
            )

            return prompt
        except FileNotFoundError:
            logger.error("RAG prompt file not found")
            raise

    async def _call_llm_with_rag_prompt(
        self,
        prompt: str
    ) -> RAGAnswer:
        """Call the LLM provider with the RAG prompt.

        Args:
            prompt: The complete RAG prompt.

        Returns:
            RAG answer from the LLM provider.
        """
        return await self.llm_provider.generate_rag_answer(
            rag_prompt=prompt
        )

    async def _generate_answer_with_context(
        self,
        question: Question,
        retrieval_result: Any
    ) -> RAGAnswer:
        """Generate an answer using LLM with retrieved context.

        Args:
            question: The user's question.
            retrieval_result: Result from vector store search.

        Returns:
            RAG answer with sources.
        """
        try:
            # Build context from retrieved chunks
            context_text = self._build_context_text(retrieval_result.chunks)

            # Determine relevant metric type based on question
            question_lower = question.text.lower()
            metric_type = "total_assets"  # Default metric

            if "asets" in question_lower:
                metric_type = "total_assets"
            elif "liabilities" in question_lower:
                metric_type = "total_liabilities"
            elif "equity" in question_lower:
                metric_type = "total_equity"
            elif "profit" in question_lower or "laba" in question_lower:
                metric_type = "net_profit"
            elif "cash" in question_lower or "kas" in question_lower:
                metric_type = "cash"
            elif "revenue" in question_lower or "pendapatan" in question_lower:
                metric_type = "revenue"

            # Aggregate pre-extracted financial data
            financial_data = self._aggregate_financial_data_from_chunks(retrieval_result.chunks, metric_type)

            # Build RAG prompt with aggregated financial data
            prompt = self._build_rag_prompt(question.text, context_text, financial_data)

            # Get LLM response
            llm_answer = await self._call_llm_with_rag_prompt(prompt)

            # Create RAG answer with the retrieved sources
            rag_answer = RAGAnswer(
                text=llm_answer.text,
                confidence_score=llm_answer.confidence_score,
                sources=retrieval_result.chunks,
                explanation=llm_answer.explanation or "Answer generated based on retrieved document excerpts"
            )

            return rag_answer
        except Exception as e:
            raise RAGServiceError(
                f"Failed to generate answer with context: {str(e)}"
            ) from e

    def initialize(
        self,
        document_paths: list[str]
    ) -> None:
        """Initialize the RAG system with documents.

        Args:
            document_paths: List of paths to PDF documents.

        Raises:
            RAGServiceError: If initialization fails.
        """
        try:
            # Process documents and extract chunks
            logger.info("Processing PDF documents...")
            chunks = self.document_processor.process_documents(document_paths)
            logger.info(f"Extracted {len(chunks)} text chunks from documents.")

            if not chunks:
                raise RAGServiceError("No text chunks were extracted from the documents.")
            
            # Add chunks to the vector store
            logger.info("Adding document chunks to the vector store...")
            self.vector_store.add_documents(chunks)

            # Mark as initialized
            self._is_initialized = True
            logger.info("RAG service initialized successfully.")
        except Exception as e:
            raise RAGServiceError(
                f"Failed to initialize RAG service: {str(e)}"
            ) from e
        
    async def process_question(
        self,
        question: Question
    ) -> RAGAnswer:
        """Process a question using the RAG pipeline.

        Args:
            question: The user's question.

        Returns:
            RAG answer based on retrieved documents.

        Raises:
            RAGServiceError: If processing fails.
        """
        if not self._is_initialized:
            raise RAGServiceError("RAG service is not initialized with documents.")

        try:
            # Retrieve relevant document chunks
            retrieval_result = self.vector_store.search(
                query=question.text,
                n_results=5
            )

            if not retrieval_result.chunks:
                # No relevant documents found
                return RAGAnswer(
                    text=f"I'm sorry, but I couldn't find relevant information in the financial documents to answer your question about: '{question.text}'.\n"
                          f"Please try rephrasing your question or asking about specific financial figures or terms that might be in the documents.",
                    confidence_score=0.1,
                    sources=[],
                    explanation="No relevant documents were retrieved."
                )

            # Generate answer using LLM
            rag_answer = await self._generate_answer_with_context(
                question=question,
                retrieval_result=retrieval_result
            )

            return rag_answer
        except Exception as e:
            raise RAGServiceError(
                f"Failed to process question: {str(e)}"
            ) from e
        
    def is_initialized(self) -> bool:
        """Check if the RAG service is initialized.

        Returns:
            True if initialized, False otherwise.
        """
        return self._is_initialized
    
    def get_financial_data_for_visualization(
        self,
        rag_answer: RAGAnswer,
        question: str
    ) -> dict:
        """Extract structured financial data from RAG answer for visualization.

        Args:
            rag_answer: The RAG answer containing source chunks.
            question: The original user question.

        Returns:
            Dictionary with financial data ready for visualization.
        """
        # Determine relevant metric type based on question
        question_lower = question.lower()
        metric_type = "total_assets"  # Default metric

        if "asets" in question_lower:
            metric_type = "total_assets"
        elif "liabilities" in question_lower:
            metric_type = "total_liabilities"
        elif "equity" in question_lower:
            metric_type = "total_equity"
        elif "profit" in question_lower or "laba" in question_lower:
            metric_type = "net_profit"
        elif "cash" in question_lower or "kas" in question_lower:
            metric_type = "cash"
        elif "revenue" in question_lower or "pendapatan" in question_lower:
            metric_type = "revenue"

        # Aggregate pre-extracted financial data from RAG answer sources
        return self._aggregate_financial_data_from_chunks(rag_answer.sources, metric_type)

    def get_status(self) -> dict[str, Any]:
        """Get the current status of the RAG service.

        Returns:
            Dictionary with status information.
        """
        status = {
            "initialized": self._is_initialized,
            "vector_store_info": {}
        }

        if hasattr(self.vector_store, 'get_collection_info'):
            status["vector_store_info"] = self.vector_store.get_collection_info()

        if hasattr(self.vector_store, 'get_document_names'):
            status["document_names"] = self.vector_store.get_document_names()

        if hasattr(self.vector_store, 'get_stats_by_document'):
            status["document_stats"] = self.vector_store.get_stats_by_document()

        return status
