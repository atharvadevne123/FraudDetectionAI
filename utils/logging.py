"""Logging configuration helpers for the fraud detection system."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
    "<level>{message}</level>"
)


def configure_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    rotation: str = "100 MB",
    retention: str = "7 days",
) -> None:
    """Configure loguru for the application.

    Removes the default handler and adds a stdout sink plus an optional
    rotating file sink. Call once at application startup.
    """
    logger.remove()
    logger.add(sys.stdout, level=level, format=LOG_FORMAT, colorize=True)
    if log_file is not None:
        logger.add(
            str(log_file),
            level=level,
            format=LOG_FORMAT,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )


def get_logger(name: str):
    """Return a loguru logger bound with the given module name."""
    return logger.bind(name=name)
