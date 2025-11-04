import os
import sys
import logging


def _setup_logging():
    """Setup logging configuration using environment variables."""
    try:
        from .config import get_config
        config = get_config()
        log_dir = config.log_dir
        log_file_name = config.log_file_name
    except (ImportError, Exception):
        # Fallback to defaults if config is not available
        log_dir = 'logs'
        log_file_name = 'rag_based_financial_report_analyzer.log'

    log_file_path = os.path.join(log_dir, log_file_name)
    os.makedirs(log_dir, exist_ok=True)

    logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

    logging.basicConfig(
        level=logging.INFO,
        format=logging_str,
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger('rag_based_financial_report_analyzer_logger')


logger = _setup_logging()