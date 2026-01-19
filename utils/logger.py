import logging
import sys
import os
from typing import Optional

from utils.config import Config

class LoggerManager:
    """Centralized logging manager for the application."""

    _instance: Optional["LoggerManager"] = None
    _logger: Optional[logging.Logger] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.initialized = True
            self._setup_logger()

    def _setup_logger(self):
        self._logger = logging.getLogger("lumenore_analytics")

        if getattr(Config, "DEBUG", False):
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.CRITICAL + 1)

        if self._logger.handlers:
            return

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s",
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

        log_file_path = os.path.join(LOG_DIR, "lumenore_analytics.log")
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

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        if self._logger is None:
            self._setup_logger()

        if name:
            return logging.getLogger(f"lumenore_analytics.{name}")
        return self._logger

    def debug(self, message: str, name: Optional[str] = None):
        logger = self.get_logger(name)
        logger.debug(message)

    def info(self, message: str, name: Optional[str] = None):
        logger = self.get_logger(name)
        logger.info(message)

    def warning(self, message: str, name: Optional[str] = None):
        logger = self.get_logger(name)
        logger.warning(message)

    def error(self, message: str, name: Optional[str] = None):
        logger = self.get_logger(name)
        logger.error(message)

    def critical(self, message: str, name: Optional[str] = None):
        logger = self.get_logger(name)
        logger.critical(message)


# Global logger instance
logger_manager = LoggerManager()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logger_manager.get_logger(name)


def debug(message: str, name: Optional[str] = None):
    logger_manager.debug(message, name)


def info(message: str, name: Optional[str] = None):
    logger_manager.info(message, name)


def warning(message: str, name: Optional[str] = None):
    logger_manager.warning(message, name)


def error(message: str, name: Optional[str] = None):
    logger_manager.error(message, name)


def critical(message: str, name: Optional[str] = None):
    logger_manager.critical(message, name)
