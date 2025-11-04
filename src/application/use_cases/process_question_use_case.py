import time
from ...domain.entities import (
    AnalysisResult,
    Question,
    VisualizationType,
    DocumentChunk,
    RAGAnswer,
    Visualization
)
from ...domain.repositories import (
    ChartGenerationError,
    IChartGenerator,
    IRAGService
)


class ProcessQuestionUseCase:
    """Use case for processing user questions using RAG.

    This class orchestrates the flow of data between different components
    to analyze user questions using Retrieval-Augmented Generation
    and generate comprehensive results with document sources.
    """

    def __init__(
        self,
        rag_service: IRAGService,
        chart_generator: IChartGenerator,
    ):
        """Initialize the use case with required dependencies.

        Args:
            llm_provider: Service for LLM interactions.
            data_repository: Service for data access.
            code_executor: Service for code execution.
            chart_generator: Service for visualization generation.
            prompt_service: Service for prompt building.
        """
        self.rag_service = rag_service
        self.chart_generator = chart_generator

    def _suggest_chart_type(self, question_text: str) -> VisualizationType:
        """Suggest the best chart type based on the question.

        Args:
            question_text: The user's question.

        Returns:
            Suggested visualization type.
        """
        question_text_lower = question_text.lower()

        if any(keyword in question_text_lower for keyword in ["trend", "over time", "timeline", "series"]):
            return VisualizationType.LINE
        elif any(keyword in question_text_lower for keyword in ["compare", "vs", "versus", "difference"]):
            return VisualizationType.BAR
        elif any(keyword in question_text_lower for keyword in ["proportion", "percentage", "share", "distribution"]):
            return VisualizationType.PIE
        else:
            return VisualizationType.BAR  # Default choice

    async def execute(self, question_text: str) -> AnalysisResult:
        """Execute the question processing use case using RAG.

        Args:
            question_text: The user's natural language question.

        Returns:
            AnalysisResult containing the complete RAG analysis.

        Raises:
            Various exceptions from the underlying services.
        """
        start_time = time.time()

        # Create Question entity
        question = Question(
            text=question_text,
            language="en"
        ) # Auto-detect language can be implemented later

        try:
            # Process question using RAG service
            rag_answer = await self.rag_service.process_question(question)

            # Get structured financial data for visualization
            financial_data = self.rag_service.get_financial_data_for_visualization(rag_answer, question_text)

            # Generate visualizations based on structured financial data
            visualizations = None
            if financial_data and financial_data.get("periods"):  # Only generate if we have financial data
                try:
                    # Auto-detect best visualization type
                    chart_type = self._suggest_chart_type(question_text)
                    visualizations = self.chart_generator.generate_chart(
                        chart_type=chart_type,
                        data=financial_data,
                        title=f"Financial Analysis: {financial_data.get('metric', 'Financial Data')}",
                        config={
                            "description": rag_answer.text,
                            "sources": [chunk.document_name for chunk in rag_answer.sources],
                            "currency": financial_data.get("currency", "IDR")
                        }
                    )
                except ChartGenerationError:
                    pass

            execution_time = time.time() - start_time

            return AnalysisResult(
                question=question,
                rag_answer=rag_answer,
                visualization=visualizations,
                execution_time=execution_time
            )
        except Exception as e:
            # Return a result with error information
            execution_time = time.time() - start_time

            # Create error RAG answer
            error_rag_answer = RAGAnswer(
                text=f"An error occurred while processing your question: {str(e)}",
                confidence_score=0.0,
                sources=[],
                explanation="Processing failed due to an error."
            )

            return AnalysisResult(
                question=question,
                rag_answer=error_rag_answer,
                visualization=None,
                execution_time=execution_time
            )