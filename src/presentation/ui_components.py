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
        st.write(result.answer.text)

        if result.data_summary.error_message:
            with st.expander("🔍 Technical Details"):
                st.code(result.data_summary.error_message)

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
        data_summary
    ) -> None:
        """Render the source documents component.

        Args:
            data_summary: The data summary entity containing source information.
        """
        st.subheader("📄 Source Documents")
        st.write(data_summary.summary_text)

        if data_summary.data and "sources" in data_summary.data:
            sources = data_summary.data["sources"]

            if sources:
                for i, source in enumerate(sources, 1):
                    with st.expander(f"📖 Source {i}: {source['document_name']}", expanded=i <= 2):
                        st.markdown(f"**Document:** {source['document_name']}")
                        if source['page_number']:
                            st.markdown(f"**Page:** {source['page_number']}")
                        st.markdown("**Content Preview")
                        st.info(source['content_preview'])
            else:
                st.info("No source documents were retrieved for this question.")
        else:
            st.info("No source document information available.")\

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
            if result.data_summary.execution_successful:
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
                "Show the 5 cities with the highest total profit",
                "Create a bar chart of total GMV per region",
                "What is the average profit per transaction for each customer segment?",
                "How is the trend of quantity of goods sold per month in 2014?",
                "Compare total GMV between 'Furniture' and 'Technology' categories",
                "What sub-categories are within the 'Office Supplies' category?"
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
        """Render the analysis result including answer, data summary, and visualizations.

        Args:
            analysis_result: The complete analysis result to display.
        """
        if not analysis_result.data_summary.execution_successful:
            UIComponents._render_error(analysis_result)

        # Create three columns for the main results
        col1, col2, col3 = st.columns(3)

        with col1:
            UIComponents._render_ai_answer(analysis_result.answer)
        with col2:
            UIComponents._render_source_documents(analysis_result.data_summary)
        with col3:
            if analysis_result.visualization:
                UIComponents._render_visualization(analysis_result.visualization)

        # Render execution info at the bottom
        UIComponents._render_execution_info(analysis_result)
