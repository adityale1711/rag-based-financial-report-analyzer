"""Centralized configuration management with validation."""

import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

try:
    import streamlit.runtime.scriptrunner as scriptrunner
    import streamlit.secrets as secrets
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


@dataclass
class Config:
    """Centralized configuration class with validation."""

    # Database Configuration
    persist_directory: str
    chroma_db_collection_name: str

    # File Paths
    rag_answer_prompt_path: str
    rag_prompt_path: str
    log_dir: str
    log_file_name: str
    document_paths: List[str]

    # OpenAI Configuration
    openai_api_key: str
    llm_model: str
    embedding_model: str
    llm_temperature: float
    max_completion_tokens: int
    max_tokens: int

    # RAG Configuration
    default_confidence_score: float
    min_confidence_score: float

    # URL-based Data Loading (optional fields at the end)
    data_url: Optional[str] = None
    zip_password: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate all configuration values."""
        # Validate required fields
        required_fields = [
            'openai_api_key', 'persist_directory', 'rag_answer_prompt_path',
            'rag_prompt_path', 'log_dir', 'log_file_name'
        ]

        for field in required_fields:
            value = getattr(self, field)
            if not value:
                raise ValueError(f"Required configuration field '{field}' is missing or empty")

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

        # Validate paths exist
        self._validate_paths()

    def _validate_paths(self) -> None:
        """Validate that required paths and files exist."""
        # Check if prompt files exist
        if not os.path.exists(self.rag_answer_prompt_path):
            raise ValueError(f"RAG answer prompt file not found: {self.rag_answer_prompt_path}")

        if not os.path.exists(self.rag_prompt_path):
            raise ValueError(f"RAG prompt file not found: {self.rag_prompt_path}")

        # Check if document directory exists
        if self.document_paths:
            for doc_path in self.document_paths:
                if not os.path.exists(doc_path):
                    raise ValueError(f"Document file not found: {doc_path}")

        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

    @property
    def log_file_path(self) -> str:
        """Get full log file path."""
        return os.path.join(self.log_dir, self.log_file_name)


class ConfigLoader:
    """Configuration loader with environment variable support."""

    @staticmethod
    def _get_env_value(key: str, default: str = '') -> str:
        """Get configuration value from environment or Streamlit secrets."""
        # Try Streamlit secrets first (for deployment)
        if STREAMLIT_AVAILABLE:
            try:
                if hasattr(secrets, key):
                    return secrets[key]
                # Also check nested keys in secrets
                for secret_key, secret_value in secrets.items():
                    if isinstance(secret_value, dict) and key in secret_value:
                        return secret_value[key]
            except Exception:
                pass  # Fallback to environment variables

        # Fall back to environment variables (for local development)
        return os.getenv(key, default)

    @staticmethod
    def load() -> Config:
        """Load configuration from environment variables or Streamlit secrets."""
        # Load environment variables from .env file for local development
        load_dotenv()

        # Parse document paths from comma-separated string
        document_paths_str = ConfigLoader._get_env_value('DOCUMENT_PATHS', '')
        document_paths = [path.strip() for path in document_paths_str.split(',') if path.strip()]

        return Config(
            # Database Configuration
            persist_directory=ConfigLoader._get_env_value('PERSIST_DIRECTORY', ''),
            chroma_db_collection_name=ConfigLoader._get_env_value('CHROMA_DB_COLLECTION_NAME', 'financial_documents'),

            # File Paths
            rag_answer_prompt_path=ConfigLoader._get_env_value('RAG_ANSWER_PROMPT_PATH', ''),
            rag_prompt_path=ConfigLoader._get_env_value('RAG_PROMPT_PATH', ''),
            log_dir=ConfigLoader._get_env_value('LOG_DIR', ''),
            log_file_name=ConfigLoader._get_env_value('LOG_FILE_NAME', ''),
            document_paths=document_paths,

            # OpenAI Configuration
            openai_api_key=ConfigLoader._get_env_value('OPENAI_API_KEY', ''),
            llm_model=ConfigLoader._get_env_value('LLM_MODEL', 'gpt-4o'),
            embedding_model=ConfigLoader._get_env_value('EMBEDDING_MODEL', 'text-embedding-3-small'),
            llm_temperature=float(ConfigLoader._get_env_value('LLM_TEMPERATURE', '0.1')),
            max_completion_tokens=int(ConfigLoader._get_env_value('MAX_COMPLETION_TOKENS', '2000')),
            max_tokens=int(ConfigLoader._get_env_value('MAX_TOKENS', '2000')),

            # RAG Configuration
            default_confidence_score=float(ConfigLoader._get_env_value('DEFAULT_CONFIDENCE_SCORE', '0.85')),
            min_confidence_score=float(ConfigLoader._get_env_value('MIN_CONFIDENCE_SCORE', '0.1')),

            # URL-based Data Loading
            data_url=ConfigLoader._get_env_value('DATA_URL') or None,
            zip_password=ConfigLoader._get_env_value('ZIP_PASSWORD') or None
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