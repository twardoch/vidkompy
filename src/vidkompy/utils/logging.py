#!/usr/bin/env python3
# this_file: src/vidkompy/utils/logging.py

"""
Logging utilities and constants for vidkompy.

This module provides standard logging formats and configuration
used across the application.
"""

import sys

# Standard logging formats
FORMAT_VERBOSE = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

FORMAT_COMPACT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)


def make_logger(name: str, verbose: bool = False) -> "Logger":
    """
    Create a logger with standardized configuration.

    Centralizes logger wiring to ensure consistent formatting
    and configuration across the application.

    Args:
        name: Logger name, typically __name__
        verbose: Whether to use verbose formatting with file/line info

    Returns:
        Configured logger instance

    """
    from loguru import logger

    # Remove default handler and add our custom one
    logger.remove()
    format_str = FORMAT_VERBOSE if verbose else FORMAT_COMPACT
    logger.add(sys.stderr, format=format_str, level="DEBUG" if verbose else "INFO")

    return logger.bind(name=name)


__all__ = ["FORMAT_COMPACT", "FORMAT_VERBOSE", "make_logger"]
