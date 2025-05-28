#!/usr/bin/env python3
# this_file: src/vidkompy/utils/__init__.py

"""
Utility modules for vidkompy.

Re-exports commonly used functions from correlation and image modules
to simplify import paths throughout the codebase.

Note: Logging utilities are now accessed directly from vidkompy.utils.logging
to keep utilities I/O-free.
"""

from vidkompy.utils.numba_ops import (
    compute_normalized_correlation,
    histogram_correlation,
    CORR_EPS,
)
from vidkompy.utils.image import ensure_gray, resize_frame

__all__ = [
    "CORR_EPS",
    "compute_normalized_correlation",
    "ensure_gray",
    "histogram_correlation",
    "resize_frame",
]
