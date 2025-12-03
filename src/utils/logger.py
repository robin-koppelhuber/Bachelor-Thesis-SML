"""Logging utilities"""

import logging
import sys
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output"""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[96m",  # Cyan
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[91m\033[1m",  # Red + Bold
    }
    RESET = "\033[0m"

    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    console_format: str = "simple",
    name: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for file logging
        console_format: Console format style (simple, detailed, json)
        name: Logger name (None for root logger)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    # Console format
    if console_format == "simple":
        console_fmt = "%(levelname)s: %(message)s"
    elif console_format == "detailed":
        console_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    elif console_format == "json":
        console_fmt = (
            '{"time": "%(asctime)s", "name": "%(name)s", '
            '"level": "%(levelname)s", "message": "%(message)s"}'
        )
    else:
        console_fmt = "%(levelname)s: %(message)s"

    console_formatter = ColoredFormatter(console_fmt)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file

        # Detailed format for file
        file_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        file_formatter = logging.Formatter(file_fmt)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger by name

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
