import logging
import sys
import os
from typing import Optional

from utils.config import Config

class LoggerManager:
    """Centralized logging manager with module prefixes."""

    _instance: Optional["LoggerManager"] = None
    _logger: Optional[logging.Logger] = None
    _module_loggers: dict = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.initialized = True
            self._setup_logger()

    def _setup_logger(self):
        self._logger = logging.getLogger("rag")

        if getattr(Config, "DEBUG", False):
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.CRITICAL + 1)

        if self._logger.handlers:
            return

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stderr)
        if getattr(Config, "DEBUG", False):
            console_handler.setLevel(logging.DEBUG)
        else:
            console_handler.setLevel(logging.CRITICAL + 1)
        console_handler.setFormatter(formatter)

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        LOG_DIR = os.path.join(BASE_DIR, ".logs")
        os.makedirs(LOG_DIR, exist_ok=True)

        log_file_path = os.path.join(LOG_DIR, "rag.log")
        file_handler = logging.FileHandler(log_file_path)
        if getattr(Config, "DEBUG", False):
            file_handler.setLevel(logging.INFO)
        else:
            file_handler.setLevel(logging.CRITICAL + 1)
        file_handler.setFormatter(formatter)

        self._logger.addHandler(console_handler)
        self._logger.addHandler(file_handler)

        self._logger.propagate = False

        self._logger.info("Logger initialized")
        self._logger.info(f"Debug mode: {getattr(Config, 'DEBUG', False)}")

    def get_logger(
        self,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> logging.Logger:
        """
        Get a logger with optional prefix/suffix.
        
        Args:
            name: Module name (e.g., 'vector_store')
            prefix: Prefix for logs (e.g., '[UI]', '[DB]', '[API]')
            suffix: Suffix for logs (e.g., '[WORKER]', '[SYNC]')
        
        Examples:
            get_logger()  # Default root logger
            get_logger(name="embedding")  # rag.embedding logger
            get_logger(prefix="[UI]")  # Root logger with [UI] prefix
            get_logger(name="db", prefix="[DATABASE]")  # rag.db with prefix
            get_logger(name="tasks", prefix="[CELERY]", suffix="[ASYNC]")
        """
        if self._logger is None:
            self._setup_logger()

        cache_key = (name, prefix, suffix)
        
        if cache_key in self._module_loggers:
            return self._module_loggers[cache_key]

        if name:
            base_name = f"rag.{name}"
        else:
            base_name = "rag"

        logger = logging.getLogger(base_name)

        if prefix or suffix:
            logger = self._wrap_logger_with_prefix_suffix(
                logger, prefix, suffix
            )

        self._module_loggers[cache_key] = logger
        return logger

    def _wrap_logger_with_prefix_suffix(
        self, logger: logging.Logger, prefix: Optional[str], suffix: Optional[str]
    ) -> logging.Logger:
        """Wrap logger to add prefix/suffix to all messages."""
        
        class PrefixSuffixAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                prefix_str = prefix or ""
                suffix_str = suffix or ""
                formatted_msg = f"{prefix_str} - {msg} {suffix_str}".strip()
                return formatted_msg, kwargs

        return PrefixSuffixAdapter(logger, {})

    def debug(
        self,
        message: str,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        logger = self.get_logger(name, prefix, suffix)
        logger.debug(message)

    def info(
        self,
        message: str,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        logger = self.get_logger(name, prefix, suffix)
        logger.info(message)

    def warning(
        self,
        message: str,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        logger = self.get_logger(name, prefix, suffix)
        logger.warning(message)

    def error(
        self,
        message: str,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        logger = self.get_logger(name, prefix, suffix)
        logger.error(message)

    def critical(
        self,
        message: str,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        logger = self.get_logger(name, prefix, suffix)
        logger.critical(message)


# Global logger instance
logger_manager = LoggerManager()


def get_logger(
    name: Optional[str] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
) -> logging.Logger:
    """
    Get a logger with optional prefix/suffix.
    
    Args:
        name: Module name (e.g., 'vector_store', 'database')
        prefix: Prefix for logs (e.g., '[UI]', '[DB]', '[API]')
        suffix: Suffix for logs (e.g., '[WORKER]', '[SYNC]')
    
    Examples:
        # Default logger
        logger = get_logger()
        
        # With module name
        logger = get_logger(name="embedding")
        
        # With prefix
        logger = get_logger(prefix="[UI]")
        
        # With both
        logger = get_logger(name="database", prefix="[DB]")
        
        # With prefix and suffix
        logger = get_logger(name="tasks", prefix="[CELERY]", suffix="[ASYNC]")
    """
    return logger_manager.get_logger(name, prefix, suffix)


def debug(
    message: str,
    name: Optional[str] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
):
    logger_manager.debug(message, name, prefix, suffix)


def info(
    message: str,
    name: Optional[str] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
):
    logger_manager.info(message, name, prefix, suffix)


def warning(
    message: str,
    name: Optional[str] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
):
    logger_manager.warning(message, name, prefix, suffix)


def error(
    message: str,
    name: Optional[str] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
):
    logger_manager.error(message, name, prefix, suffix)


def critical(
    message: str,
    name: Optional[str] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
):
    logger_manager.critical(message, name, prefix, suffix)