import re
from typing import Any
from ... import logger
from ...domain.entities import (
    AnalyzeCode,
    DocumentChunk,
    Question,
    RAGAnswer,
    Answer
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
    
    def _extract_financial_data(
        self,
        chunks: list[DocumentChunk],
        question: str
    ) -> dict:
        """Extract financial data from document chunks for visualization.

        Args:
            chunks: List of document chunks.
            question: The user's question.

        Returns:
            Dictionary with extracted financial data.
        """

        # Initialize financial data structure
        financial_data = {
            "months": [],
            "total_assets": [],
            "total_liabilities": [],
            "total_equity": [],
            "net_profit": [],
            "cash": []
        }

        # Define patterns for financial data extraction
        patterns = {
            "total_assets": r"(?:total\s+assets|aset\s+total)[\s:]*([0-9.,]+)",
            "total_liabilities": r"(?:total\s+liabilities|kewajiban\s+total)[\s:]*([0-9.,]+)",
            "total_equity": r"(?:total\s+equity|ekuitas\s+total)[\s:]*([0-9.,]+)",
            "net_profit": r"(?:net\s+profit|laba\s+bersih)[\s:]*([0-9.,]+)",
            "cash": r"(?:cash|kas)[\s:]*([0-9.,]+)"
        }

        # Month extraction pattern
        month_pattern = r"(agustus|october|oktober|november|august|oct|nov)\s+2024"

        # Extract data from chunks
        for chunk in chunks:
            content = chunk.content.lower()

            # Extract month
            month_match = re.search(month_pattern, content)
            if month_match:
                month = month_match.group(1)

                # Normalize month names
                if month in ["agustus", "august"]:
                    month = "Aug 2024"
                elif month in ["october", "oktober", "oct"]:
                    month = "Oct 2024"
                elif month in ["november", "nov"]:
                    month = "Nov 2024"

                if month not in financial_data["months"]:
                    financial_data["months"].append(month)

                    # Extract financial figures for this month
                    for key, pattern in patterns.items():
                        match = re.search(pattern, content, re.IGNORECASE)
                        if match:
                            value_str = match.group(1).replace(',', '').replace('.', '')
                            try:
                                value = float(value_str)
                                financial_data[key].append(value)
                            except ValueError:
                                financial_data[key].append(0)
                        else:
                            financial_data[key].append(0)

            # Create structured data for visualization
            if financial_data["months"]:
                # Determine which metric is most relevant to the question
                question_lower = question.lower()
                metric_key = "total_assets"  # Default metric

                if "asets" in question_lower:
                    metric_key = "total_assets"
                elif "liabilities" in question_lower:
                    metric_key = "total_liabilities"
                elif "equity" in question_lower:
                    metric_key = "total_equity"
                elif "profit" in question_lower or "laba" in question_lower:
                    metric_key = "net_profit"
                elif "cash" in question_lower or "kas" in question_lower:
                    metric_key = "cash"

                structured_data = {
                    "months": financial_data["months"],
                    "values": financial_data[metric_key],
                    "metric": metric_key.replace('_', ' ').title()
                }

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
    ) -> tuple[Answer, AnalyzeCode]:
        """Call the LLM provider with the RAG prompt.

        Args:
            prompt: The complete RAG prompt.

        Returns:
            Tuple containing RAG answer and analysis code.
        """
        return await self.llm_provider.generate_rag_answer(
            rag_prompt=prompt
        )

    async def _generate_answer_with_context(
        self,
        question: Question,
        retrieval_result: Any
    ) -> tuple[RAGAnswer, AnalyzeCode]:
        """Generate an answer using LLM with retrieved context.

        Args:
            question: The user's question.
            retrieval_result: Result from vector store search.

        Returns:
            Tuple containing RAG answer and analysis code.
        """
        try:
            # Build context from retrieved chunks
            context_text = self._build_context_text(retrieval_result.chunks)

            # Extract financial data for visualization
            financial_data = self._extract_financial_data(retrieval_result.chunks, question.text)

            # Build RAG prompt with financial data extraction instruction
            prompt = self._build_rag_prompt(question.text, context_text, financial_data)

            # Get LLM response
            llm_answer, llm_code = await self._call_llm_with_rag_prompt(prompt)

            # Convert the LLM answer to RAG answer format
            rag_answer = RAGAnswer(
                text=llm_answer.text,
                confidence_score=llm_answer.confidence_score,
                sources=retrieval_result.chunks,
                explanation=llm_answer.explanation or "Answer generated based on retrieved document excerpts"
            )

            return rag_answer, llm_code
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
    ) -> tuple[RAGAnswer, AnalyzeCode]:
        """Process a question using the RAG pipeline.

        Args:
            question: The user's question.

        Returns:
            Tuple containing RAG answer and visualization code.

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
                no_context_answer = RAGAnswer(
                    text=f"I'm sorry, but I couldn't find relevant information in the financial documents to answer your question about: '{question.text}'.\n"
                          f"Please try rephrasing your question or asking about specific financial figures or terms that might be in the documents.",
                    confidence_score=0.1,
                    sources=[],
                    explanation="No relevant document were retrieved."
                )

                no_code = AnalyzeCode(
                    code="# No visualization - no relevant data found",
                    description="No visualization generated as no relevant information was found."
                )

                return no_context_answer, no_code

            # Generate answer using LLM
            rag_answer, analysis_code = await self._generate_answer_with_context(
                question=question,
                retrieval_result=retrieval_result
            )

            return rag_answer, analysis_code
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
