#!/usr/bin/env python3
# this_file: src/vidkompy/utils/logging.py

"""
Logging utilities and constants for vidkompy.

This module provides standard logging formats and configuration
used across the application.
"""

# Standard logging formats
LOG_FORMAT_VERBOSE = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

LOG_FORMAT_DEFAULT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)

__all__ = ["LOG_FORMAT_DEFAULT", "LOG_FORMAT_VERBOSE"]
