"""Centralized configuration management with validation."""

import os
from typing import List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv


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
    def load() -> Config:
        """Load configuration from environment variables."""
        # Load environment variables from .env file
        load_dotenv()

        # Parse document paths from comma-separated string
        document_paths_str = os.getenv('DOCUMENT_PATHS', '')
        document_paths = [path.strip() for path in document_paths_str.split(',') if path.strip()]

        return Config(
            # Database Configuration
            persist_directory=os.getenv('PERSIST_DIRECTORY', ''),
            chroma_db_collection_name=os.getenv('CHROMA_DB_COLLECTION_NAME', 'financial_documents'),

            # File Paths
            rag_answer_prompt_path=os.getenv('RAG_ANSWER_PROMPT_PATH', ''),
            rag_prompt_path=os.getenv('RAG_PROMPT_PATH', ''),
            log_dir=os.getenv('LOG_DIR', ''),
            log_file_name=os.getenv('LOG_FILE_NAME', ''),
            document_paths=document_paths,

            # OpenAI Configuration
            openai_api_key=os.getenv('OPENAI_API_KEY', ''),
            llm_model=os.getenv('LLM_MODEL', 'gpt-4o'),
            embedding_model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
            llm_temperature=float(os.getenv('LLM_TEMPERATURE', '0.1')),
            max_completion_tokens=int(os.getenv('MAX_COMPLETION_TOKENS', '2000')),
            max_tokens=int(os.getenv('MAX_TOKENS', '2000')),

            # RAG Configuration
            default_confidence_score=float(os.getenv('DEFAULT_CONFIDENCE_SCORE', '0.85')),
            min_confidence_score=float(os.getenv('MIN_CONFIDENCE_SCORE', '0.1'))
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