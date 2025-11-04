import streamlit as st
from ..domain.entities import AnalysisResult, Visualization


class UIComponents:
    """Collection of reusable UI components for the application."""

    @staticmethod
    def _render_error(
        result: AnalysisResult
    ) -> None:
        """Render error information.

        Args:
            result: The analysis result containing the error.
        """
        st.error("❌ An Error Occurred")
        st.write(result.rag_answer.text)

        if result.rag_answer.explanation and "error" in result.rag_answer.explanation.lower():
            with st.expander("🔍 Technical Details"):
                st.code(result.rag_answer.explanation)

    @staticmethod
    def _render_ai_answer(
        answer
    ) -> None:
        """Render the AI answer component.

        Args:
            answer: The answer entity to display.
        """
        st.subheader("🤖 AI Answer")
        st.write(answer.text)

        if answer.confidence_score:
            st.markdown(f"**Confidence Score:** {answer.confidence_score:.1%}")

        if answer.explanation:
            with st.expander("📝 Explanation"):
                st.write(answer.explanation)

    @staticmethod
    def _render_source_documents(
        rag_answer
    ) -> None:
        """Render the source documents component.

        Args:
            rag_answer: The RAG answer containing source information.
        """
        st.subheader("📄 Source Documents")
        st.write(f"RAG analysis completed with {len(rag_answer.sources)} source documents")

        if rag_answer.sources:
            for i, chunk in enumerate(rag_answer.sources, 1):
                with st.expander(f"📖 Source {i}: {chunk.document_name}", expanded=i <= 2):
                    st.markdown(f"**Document:** {chunk.document_name}")
                    if chunk.page_number:
                        st.markdown(f"**Page:** {chunk.page_number}")
                    st.markdown("**Content Preview")
                    st.info(chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content)
        else:
            st.info("No source documents were retrieved for this question.")\

    @staticmethod
    def _render_visualization(
        visualization: Visualization
    ) -> None:
        """Render a single visualization component.

        Args:
            visualization: The visualization entity to display.
        """
        st.subheader(f"📈 Visualization")

        if hasattr(visualization.chart_object, "show"):
            st.plotly_chart(visualization.chart_object, use_container_width=True)
        else:
            st.write("Chart data format not supported for display.")

        if visualization.description:
            st.caption(visualization.description)

    @staticmethod
    def _render_execution_info(
        result: AnalysisResult
    ) -> None:
        """Render execution information.

        Args:
            result: The analysis result with execution info.
        """
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.caption(f"⏱️ Execution Time: {result.execution_time:.2f} seconds")
        with col2:
            if result.rag_answer.confidence_score > 0.1:  # Successful analysis has reasonable confidence
                st.caption("✅ Analysis completed successfully")
            else:
                st.caption("❌ Analysis failed")

    @staticmethod
    def render_header(
        title: str,
        subtitle: str = ""
    ) -> None:
        """Render the application header.

        Args:
            title: The main title of the application.
            subtitle: The subtitle or description.
        """
        st.set_page_config(
            page_title=title,
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title(title)
        if subtitle:
            st.markdown(f"*{subtitle}*")
        st.markdown("---")

    @staticmethod
    def render_question_input() -> str:
        """Render the question input component.

        Returns:
            The user's input question.
        """
        st.subheader("Ask a Question about the Data")

        # Example questions
        with st.expander("Example Questions"):
            example_questions = [
                "**BAR CHART EXAMPLES** (Comparison questions)",
                "--------------------------------------------------------------------------",
                "Compare total assets between August and November 2024",
                "Compare total liabilities between August, October, and November 2024",
                "Show comparison of total equity across all three months",
                "--------------------------------------------------------------------------",
                "**PIE CHART EXAMPLES** (Proportion/Percentage/Share questions)",
                "--------------------------------------------------------------------------",
                "What is the proportion of cash in total assets for August 2024?",
                "What percentage of total assets consists of cash in November 2024?",
                "Show the share of equity in total assets for October 2024",
                "--------------------------------------------------------------------------",
                "**LINE CHART EXAMPLES** (Trend/Timeline questions)",
                "--------------------------------------------------------------------------",
                "Show me the trend of total assets over the past three months",
                "What is the timeline of cash position changes from August to November 2024?",
                "Display the trend in total equity over time",
                "--------------------------------------------------------------------------",
                "**GENERAL ANALYSIS QUESTIONS**",
                "--------------------------------------------------------------------------",
                "What is the percentage change in total assets from August to November 2024?",
                "How has the bank's financial position evolved over the past three months?",
                "Analyze the growth in total assets across the reporting period"
            ]
            for question in example_questions:
                st.write(f"• {question}")

        question = st.text_input(
            label="Enter your question:",
            placeholder="E.g., Show the top 10 products by sales",
            key="question_input"
        )

        return question
    
    @staticmethod
    def render_analysis_result(
        analysis_result: AnalysisResult
    ) -> None:
        """Render the analysis result including RAG answer, sources, and visualizations.

        Args:
            analysis_result: The complete analysis result to display.
        """
        if analysis_result.rag_answer.confidence_score <= 0.1:  # Error condition
            UIComponents._render_error(analysis_result)

        # Create three columns for the main results
        col1, col2, col3 = st.columns(3)

        with col1:
            UIComponents._render_ai_answer(analysis_result.rag_answer)
        with col2:
            UIComponents._render_source_documents(analysis_result.rag_answer)
        with col3:
            if analysis_result.visualization:
                UIComponents._render_visualization(analysis_result.visualization)

        # Render execution info at the bottom
        UIComponents._render_execution_info(analysis_result)
