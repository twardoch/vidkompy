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
    - vidkompy/comp/align.py
    - vidkompy/comp/data_types.py
    - vidkompy/comp/vidkompy.py
    """

    # BORDER = "border"  # Deferred post-MVP
    PRECISE = "precise"  # Use frame-based matching (Tunnel Syncer based for MVP)
