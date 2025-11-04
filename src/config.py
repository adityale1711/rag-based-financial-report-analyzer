"""Centralized configuration management with validation."""

import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

try:
    import streamlit.runtime.scriptrunner as scriptrunner
    import streamlit.secrets as secrets
    STREAMLIT_AVAILABLE = True

    # Check if we're running in Streamlit Cloud by testing if secrets exist
    try:
        _ = secrets.items()
        STREAMLIT_CLOUD = True
    except Exception:
        STREAMLIT_CLOUD = False

except ImportError:
    STREAMLIT_AVAILABLE = False
    STREAMLIT_CLOUD = False
    secrets = None


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
        # Validate required fields (only API key is strictly required)
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")

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

        # Set default values for missing fields (cloud-friendly)
        if not self.persist_directory:
            self.persist_directory = "/tmp/chroma_db"
        if not self.chroma_db_collection_name:
            self.chroma_db_collection_name = "financial_documents"
        if not self.rag_answer_prompt_path:
            self.rag_answer_prompt_path = "prompts/infrastructure/external/rag_answer_prompt.txt"
        if not self.rag_prompt_path:
            self.rag_prompt_path = "prompts/application/services/rag_prompt.txt"
        if not self.log_dir:
            self.log_dir = "/tmp/logs"
        if not self.log_file_name:
            self.log_file_name = "financial_analyzer.log"

        # Validate and create directories
        self._validate_paths()

    def _validate_paths(self) -> None:
        """Validate that required paths and files exist."""# Check if prompt files exist (only for local development, not Streamlit Cloud)
        if not STREAMLIT_CLOUD:
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
        
        # Try Streamlit secrets first (for Streamlit Cloud deployment)
        if STREAMLIT_CLOUD and secrets:
            try:
                # Handle nested structure in secrets.toml
                # Check for direct key access first
                if key in secrets:
                    return secrets[key]

                # Check nested structures (e.g., [openai] api_key)
                for section_name, section_data in secrets.items():
                    if isinstance(section_data, dict) and key.lower() in section_data:
                        return section_data[key.lower()]

                    # Handle specific mappings for nested structures
                    key_mapping = {
                        'OPENAI_API_KEY': ('openai', 'api_key'),
                        'DATA_URL': ('data', 'url'),
                        'ZIP_PASSWORD': ('data', 'zip_password'),
                        'PERSIST_DIRECTORY': ('paths', 'persist_directory'),
                        'CHROMA_DB_COLLECTION_NAME': ('paths', 'chroma_db_collection_name') or ('chroma', 'collection_name'),
                        'RAG_ANSWER_PROMPT_PATH': ('paths', 'rag_answer_prompt_path'),
                        'RAG_PROMPT_PATH': ('paths', 'rag_prompt_path'),
                        'LOG_DIR': ('paths', 'log_dir'),
                        'LOG_FILE_NAME': ('paths', 'log_file_name'),
                        'LLM_MODEL': ('llm', 'model'),
                        'LLM_TEMPERATURE': ('llm', 'temperature'),
                        'MAX_COMPLETION_TOKENS': ('llm', 'max_completion_tokens'),
                        'MAX_TOKENS': ('llm', 'max_tokens'),
                        'EMBEDDING_MODEL': ('embedding', 'model'),
                        'DEFAULT_CONFIDENCE_SCORE': ('rag', 'default_confidence_score'),
                        'MIN_CONFIDENCE_SCORE': ('rag', 'min_confidence_score')
                    }

                    if key in key_mapping:
                        section, nested_key = key_mapping[key]
                        if section_name == section and isinstance(section_data, dict) and nested_key in section_data:
                            return section_data[nested_key]

            except Exception as e:
                # Log the error for debugging but continue to fallback
                print(f"Warning: Failed to read Streamlit secrets for key '{key}': {e}")
                pass  # Fallback to environment variables

        # Fall back to environment variables (for local development)
        env_value = os.getenv(key, default)
        return env_value

    @staticmethod
    def get_config_source() -> str:
        """Get the current configuration source being used."""
        if STREAMLIT_CLOUD and secrets:
            try:
                _ = secrets.items()  # Test if secrets are accessible
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