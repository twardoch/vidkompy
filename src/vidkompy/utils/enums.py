#!/usr/bin/env python3
# this_file: src/vidkompy/comp/enums.py

"""
Enumerations for vidkompy composition module.

Contains all enum types used across the composition system.
"""

from enum import Enum

__all__ = ["TimeMode"]


class TimeMode(Enum):
    """Temporal alignment modes.

    Used in:
    - vidkompy/comp/alignment_engine.py
    - vidkompy/comp/vidkompy_models.py
    """

    BORDER = "border"  # Border-based matching (default)
    PRECISE = "precise"  # Use frame-based matching
