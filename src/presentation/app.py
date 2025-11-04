import os
import asyncio
import streamlit as st
import time
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from .ui_components import UIComponents
from ..application.services.rag_service import RAGService
from ..infrastructure.external.openai_client import OpenAIClient
from ..infrastructure.visualization.plotly_chart import PlotlyChartGenerator
from ..infrastructure.vector_store.chromadb_store import ChromaDBStore
from ..application.use_cases.process_question_use_case import ProcessQuestionUseCase
from ..infrastructure.document_processing.pdf_processor import PDFProcessor
from ..infrastructure.document_processing.url_data_loader import URLDataLoader
from ..config import get_config, ConfigLoader


class StreamlitApp:
    """Main Streamlit application for the RAG-based financial report analyzer."""

    def __init__(self):
        """Initialize the Streamlit application."""
        config = get_config()

        # Initialize services and use case
        self.llm_client = OpenAIClient()
        self.document_processor = PDFProcessor()
        self.vector_store = ChromaDBStore()
        self.chart_generator = PlotlyChartGenerator()

        # Initialize RAG Service
        self.rag_service = RAGService(
            document_processor=self.document_processor,
            vector_store=self.vector_store,
            llm_provider=self.llm_client
        )

        self.process_question_use_case = ProcessQuestionUseCase(
            rag_service=self.rag_service,
            chart_generator=self.chart_generator
        )

        # Store document paths from config (will handle URL loading if needed)
        self.document_paths = config.document_paths

    def _initialize_rag_system(self) -> None:
        """Initialize the RAG system with financial documents."""
        if not self.rag_service.is_initialized():
            with st.spinner("🔄 Initializing RAG system with financial documents..."):
                try:
                    # Use thread-based timeout instead of signal
                    def load_documents_with_timeout():
                        """Load documents within a separate thread."""
                        return URLDataLoader.get_document_paths()

                    # Execute document loading with timeout using ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(load_documents_with_timeout)
                        try:
                            document_paths = future.result(timeout=30)  # 30-second timeout
                        except FutureTimeoutError:
                            st.error("❌ Document loading timed out. This might be due to network issues or large file downloads.")
                            st.info("💡 Try refreshing the page or check your internet connection.")
                            return

                    # Check if documents exist
                    existing_docs = []
                    for path in document_paths:
                        if os.path.exists(path):
                            existing_docs.append(path)
                        else:
                            st.warning(f"⚠️ Document not found: {path}")

                    if not existing_docs:
                        st.error("❌ No financial documents found. \n" \
                        "Please ensure PDF documents are placed in the configured document paths or check the DATA_URL configuration.")
                        return

                    # Initialize RAG Service
                    self.rag_service.initialize(existing_docs)

                    # Show initialization success
                    st.success(f"✅ RAG system initialized successfully with {len(existing_docs)} financial documents!")

                    # Show document stats
                    status = self.rag_service.get_status()
                    if "document_stats" in status:
                        st.info(f"📊 Processed chunks: {status['document_stats']}")
                except Exception as e:
                    st.error(f"❌ Failed to initialize RAG system: {str(e)}")
                    return

    def run(self) -> None:
        """Run the Streamlit application."""

        # Render header
        UIComponents.render_header(
            title="RAG-Based Financial Report Analyzer",
            subtitle="Ask questions about Bank Central Asia financial reports using natural language and get instant insights with source documents!"
        )

        # Check the API key
        config = get_config()
        if not config.openai_api_key:
            st.error("❌ OpenAI API key is not configured.")
            st.info("🔧 To set up the API key:")

            if ConfigLoader.get_config_source() == "Streamlit Cloud Secrets":
                st.code("""
# OpenAI Configuration
OPENAI_API_KEY = "your-openai-api-key-here"
                """)
                st.info("💡 Add this to your Streamlit Cloud secrets.toml file")
            else:
                st.code("""
# Copy .env.example to .env
OPENAI_API_KEY=your-openai-api-key-here
                """)
                st.info("💡 Add your OpenAI API key to the .env file")

            st.error("🚫 Cannot proceed without an OpenAI API key")
            return
        
        # Initialize RAG system with documents
        self._initialize_rag_system()
        
        # Initialize session state for results
        if "analysis_result" not in st.session_state:
            st.session_state.analysis_result = None

        # Render question input
        question = UIComponents.render_question_input()

        # Submit question button
        if st.button("Submit Question", type="primary", disabled=not question.strip()):
            if question.strip():
                # Clear previous result
                st.session_state.analysis_result = None

                # Show loading indicator
                loading_placeholder = st.empty()
                loading_placeholder.info("⏳ Processing your question, please wait...")

                try:
                    # Process the question asynchronously
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(
                        self.process_question_use_case.execute(question)
                    )
                    loop.close()

                    # Store the result in session state
                    st.session_state.analysis_result = result
                except Exception as e:
                    st.error(f"An error occurred while processing your question: {str(e)}")
                finally:
                    loading_placeholder.empty()
            else:
                st.warning("⚠️ Please enter a question before analyzing.")
        
        # Display results if available
        if st.session_state.analysis_result:
            UIComponents.render_analysis_result(st.session_state.analysis_result)


def main() -> None:
    """Entry point for the Streamlit application."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
