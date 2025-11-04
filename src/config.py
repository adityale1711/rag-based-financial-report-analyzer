"""Simplified centralized configuration management with flat key structure."""

import os
from typing import List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None


@dataclass
class Config:
    """Simplified configuration class with flat key structure."""

    # OpenAI Configuration
    openai_api_key: str
    llm_model: str
    embedding_model: str
    llm_temperature: float
    max_completion_tokens: int
    max_tokens: int

    # Database Configuration
    persist_directory: str
    chroma_db_collection_name: str

    # File Paths
    rag_answer_prompt_path: str
    rag_prompt_path: str
    log_dir: str
    log_file_name: str
    document_paths: List[str]

    # URL-based Data Loading
    data_url: Optional[str] = None
    zip_password: Optional[str] = None

    # RAG Configuration
    default_confidence_score: float
    min_confidence_score: float

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate all configuration values."""
        # Validate required fields
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Please set OPENAI_API_KEY in your .env file or Streamlit secrets."
            )

        # Validate numeric ranges
        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ValueError(f"LLM temperature must be between 0.0 and 2.0, got {self.llm_temperature}")

        if not 0.0 <= self.default_confidence_score <= 1.0:
            raise ValueError(f"Default confidence score must be between 0.0 and 1.0, got {self.default_confidence_score}")

        if not 0.0 <= self.min_confidence_score <= 1.0:
            raise ValueError(f"Min confidence score must be between 0.0 and 1.0, got {self.min_confidence_score}")

        if self.max_completion_tokens <= 0:
            raise ValueError(f"Max completion tokens must be positive, got {self.max_completion_tokens}")

        if self.max_tokens <= 0:
            raise ValueError(f"Max tokens must be positive, got {self.max_tokens}")

        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

    @property
    def log_file_path(self) -> str:
        """Get full log file path."""
        return os.path.join(self.log_dir, self.log_file_name)


class ConfigLoader:
    """Simplified configuration loader with flat key structure."""

    @staticmethod
    def _get_config_value(key: str, default=None):
        """Get configuration value from Streamlit secrets or environment variables.

        Args:
            key: Configuration key
            default: Default value if not found

        Returns:
            Configuration value
        """
        # Try Streamlit secrets first (for deployment)
        if STREAMLIT_AVAILABLE:
            try:
                value = st.secrets.get(key)
                if value is not None:
                    return value
            except Exception:
                pass

        # Fall back to environment variables (for local development)
        return os.getenv(key, default)

    @staticmethod
    def get_config_source() -> str:
        """Get the current configuration source being used."""
        if STREAMLIT_AVAILABLE:
            try:
                # Test if we can access secrets
                secrets_dict = dict(st.secrets.items())
                if secrets_dict:
                    return "Streamlit Cloud Secrets"
            except Exception:
                pass
        return "Environment Variables (.env)"

    @staticmethod
    def load() -> Config:
        """Load configuration from environment variables or Streamlit secrets."""
        # Load environment variables from .env file for local development
        load_dotenv()

        # Parse document paths from comma-separated string
        document_paths_str = ConfigLoader._get_config_value('DOCUMENT_PATHS', '')
        document_paths = [path.strip() for path in document_paths_str.split(',') if path.strip()]

        return Config(
            # OpenAI Configuration
            openai_api_key=ConfigLoader._get_config_value('OPENAI_API_KEY', ''),
            llm_model=ConfigLoader._get_config_value('LLM_MODEL', 'gpt-4o'),
            embedding_model=ConfigLoader._get_config_value('EMBEDDING_MODEL', 'text-embedding-3-small'),
            llm_temperature=float(ConfigLoader._get_config_value('LLM_TEMPERATURE', '0.1')),
            max_completion_tokens=int(ConfigLoader._get_config_value('MAX_COMPLETION_TOKENS', '2000')),
            max_tokens=int(ConfigLoader._get_config_value('MAX_TOKENS', '2000')),

            # Database Configuration
            persist_directory=ConfigLoader._get_config_value('PERSIST_DIRECTORY', './data/chroma_db'),
            chroma_db_collection_name=ConfigLoader._get_config_value('CHROMA_DB_COLLECTION_NAME', 'financial_documents'),

            # File Paths
            rag_answer_prompt_path=ConfigLoader._get_config_value('RAG_ANSWER_PROMPT_PATH', './prompts/infrastructure/external/rag_answer_prompt.txt'),
            rag_prompt_path=ConfigLoader._get_config_value('RAG_PROMPT_PATH', './prompts/application/services/rag_prompt.txt'),
            log_dir=ConfigLoader._get_config_value('LOG_DIR', './logs'),
            log_file_name=ConfigLoader._get_config_value('LOG_FILE_NAME', 'financial_analyzer.log'),
            document_paths=document_paths,

            # URL-based Data Loading
            data_url=ConfigLoader._get_config_value('DATA_URL') or None,
            zip_password=ConfigLoader._get_config_value('ZIP_PASSWORD') or None,

            # RAG Configuration
            default_confidence_score=float(ConfigLoader._get_config_value('DEFAULT_CONFIDENCE_SCORE', '0.85')),
            min_confidence_score=float(ConfigLoader._get_config_value('MIN_CONFIDENCE_SCORE', '0.1'))
        )


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = ConfigLoader.load()
    return _config


def reload_config() -> Config:
    """Reload configuration from environment variables."""
    global _config
    _config = ConfigLoader.load()
    return _config