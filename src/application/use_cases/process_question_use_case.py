import time
from ...domain.entities import (
    AnalysisResult, 
    Question, 
    VisualizationType,
    Answer,
    DataSummary
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
            rag_answer, analysis_code = await self.rag_service.process_question(question)

            # Create data summary for RAG results
            data_summary = DataSummary(
                data={
                    "sources": [
                        {
                            "document_name": chunk.document_name,
                            "page_numner": chunk.page_number,
                            "content_preview": chunk.content[:200] + "..." if len(chunk) > 200 else chunk.content
                        }
                        for chunk in rag_answer.sources
                    ],
                    "answer_text": rag_answer.text,
                    "confidence_score": rag_answer.confidence_score
                },
                summary_text=f"RAG analysis completed with {len(rag_answer.sources)} source documents",
                execution_successful=True,
                error_message=None
            )

            # Generate visualizations if code execution was successful
            visualizations = None
            if analysis_code and analysis_code.code and not analysis_code.code.startswith("# No"):
                try:
                    # Auto-detect best visualization type
                    chart_type = self._suggest_chart_type(question_text)
                    visualizations = self.chart_generator.generate_chart(
                        chart_type=chart_type,
                        data=data_summary.data,
                        title=f"Financial Analysis: {question_text[:50]}...",
                        config={
                            "description": rag_answer.text,
                            "sources": [chunk.document_name for chunk in rag_answer.sources]
                        }
                    )
                except ChartGenerationError:
                    pass

            execution_time = time.time() - start_time

            # Convert RAGAnswer to Answer for compatibility
            answer = Answer(
                text=rag_answer.text,
                confidence_score=rag_answer.confidence_score,
                explanation=rag_answer.explanation
            )

            return AnalysisResult(
                question=question,
                answer=answer,
                data_summary=data_summary,
                visualization=visualizations,
                execution_time=execution_time
            )
        except Exception as e:
            # Return a result with error information
            execution_time = time.time() - start_time

            error_answer = Answer(
                text=f"An error occurred while processing your question: {str(e)}",
                confidence_score=0.0,
                explanation="Processing failed due to an error."
            )

            error_summary = DataSummary(
                data=None,
                summary_text="Processing failed",
                execution_successful=False,
                error_message=str(e)
            )

            return AnalysisResult(
                question=question,
                answer=error_answer,
                data_summary=error_summary,
                visualization=None,
                execution_time=execution_time
            )