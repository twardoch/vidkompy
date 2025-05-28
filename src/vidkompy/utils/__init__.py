#!/usr/bin/env python3
# this_file: src/vidkompy/utils/__init__.py

"""
Utility modules for vidkompy.

Re-exports commonly used functions from correlation and image modules
to simplify import paths throughout the codebase.
"""

from .correlation import compute_normalized_correlation, histogram_correlation
from .image import ensure_gray, resize_frame
from .logging import LOG_FORMAT_VERBOSE, LOG_FORMAT_DEFAULT

__all__ = [
    "LOG_FORMAT_DEFAULT",
    "LOG_FORMAT_VERBOSE",
    "compute_normalized_correlation",
    "ensure_gray",
    "histogram_correlation",
    "resize_frame",
]
