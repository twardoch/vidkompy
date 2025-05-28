#!/usr/bin/env python3
# this_file: src/vidkompy/comp/enums.py

"""
Enumerations for vidkompy composition module.

Contains all enum types used across the composition system.
"""

from enum import Enum

__all__ = ["MatchTimeMode", "SpatialMethod", "TemporalMethod"]


class MatchTimeMode(Enum):
    """Temporal alignment modes.

    Used in:
    - vidkompy/comp/alignment_engine.py
    - vidkompy/comp/domain_models.py
    """

    BORDER = "border"  # Border-based matching (default)
    PRECISE = "precise"  # Use frame-based matching


class SpatialMethod(Enum):
    """Spatial alignment methods."""

    TEMPLATE = "template"  # Template matching (only method)
    CENTER = "center"  # Simple center alignment (fallback)


class TemporalMethod(Enum):
    """Temporal alignment algorithm methods.

    Used in:
    - vidkompy/comp/alignment_engine.py
    """

    DTW = "dtw"  # Dynamic Time Warping (new default)
    CLASSIC = "classic"  # Original keyframe matching
    FRAMES = "frames"  # Legacy alias for classic
