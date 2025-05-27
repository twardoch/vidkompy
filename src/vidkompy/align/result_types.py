#!/usr/bin/env python3
# this_file: src/vidkompy/align/result_types.py

"""
Data structures and types for thumbnail detection results.

This module contains all the dataclasses, enums, and type definitions
used throughout the thumbnail finder system.
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np


class PrecisionLevel(Enum):
    """Precision levels for thumbnail detection analysis.

    Used in:
    - vidkompy/align/__init__.py
    - vidkompy/align/display.py
    - vidkompy/align/precision.py
    """

    BALLPARK = 0  # Ultra-fast ballpark (~1ms) - histogram correlation
    COARSE = 1  # Coarse template matching (~10ms) - wide scale steps
    BALANCED = 2  # Balanced feature + template (~25ms) - default
    FINE = 3  # Fine feature + focused template (~50ms) - high quality
    PRECISE = 4  # Precise sub-pixel refinement (~200ms) - maximum accuracy

    @property
    def description(self) -> str:
        """Get human-readable description of the precision level.

        Used in:
        - vidkompy/align/display.py
        """
        descriptions = {
            self.BALLPARK: "Ballpark",
            self.COARSE: "Coarse",
            self.BALANCED: "Balanced",
            self.FINE: "Fine",
            self.PRECISE: "Precise",
        }
        return descriptions[self]

    @property
    def timing_estimate(self) -> str:
        """Get timing estimate for this precision level.

        Used in:
        - vidkompy/align/display.py
        """
        timings = {
            self.BALLPARK: "~1ms",
            self.COARSE: "~10ms",
            self.BALANCED: "~25ms",
            self.FINE: "~50ms",
            self.PRECISE: "~200ms",
        }
        return timings[self]


@dataclass(frozen=True)
class MatchResult:
    """
    Result of a single template matching operation.

    Attributes:
        confidence: Match confidence score (0.0 to 1.0)
        x: X coordinate of match position
        y: Y coordinate of match position
        scale: Scale factor applied to foreground
        frame_idx: Index of the frame that produced this result
        bg_frame_idx: Index of the background frame used
        method: Algorithm method used for matching
        processing_time: Time taken for the matching operation

    Used in:
    - vidkompy/align/__init__.py
    - vidkompy/align/algorithms.py
    - vidkompy/align/core.py
    - vidkompy/align/precision.py
    """

    confidence: float
    x: int
    y: int
    scale: float
    frame_idx: int = 0
    bg_frame_idx: int = 0
    method: str = "template"
    processing_time: float = 0.0

    def __post_init__(self):
        """Validate the match result data."""
        if not 0.0 <= self.confidence <= 1.0:
            msg = f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            raise ValueError(msg)
        if self.scale <= 0.0:
            msg = f"Scale must be positive, got {self.scale}"
            raise ValueError(msg)


@dataclass(frozen=True)
class PrecisionAnalysisResult:
    """
    Result from a single precision level analysis.

    Attributes:
        level: The precision level used
        scale: Detected scale factor
        x: X position
        y: Y position
        confidence: Match confidence
        processing_time: Time taken for this analysis
        method: Algorithm method used

    Used in:
    - vidkompy/align/__init__.py
    - vidkompy/align/display.py
    - vidkompy/align/precision.py
    """

    level: PrecisionLevel
    scale: float
    x: int
    y: int
    confidence: float
    processing_time: float
    method: str = "unknown"


@dataclass
class AnalysisData:
    """
    Complete analysis data for alternative result reporting.

    Attributes:
        unity_scale_result: Best result from exact 100% scale search
        scaled_result: Best result from multi-scale search
        total_results: Total number of results analyzed
        unity_scale_count: Number of near-unity scale results
        precision_level: Precision level used
        unity_scale_preference_active: Whether unity scale preference applied
        precision_analysis: List of results from each precision level

    Used in:
    - vidkompy/align/__init__.py
    - vidkompy/align/core.py
    - vidkompy/align/display.py
    """

    unity_scale_result: MatchResult | None = None
    scaled_result: MatchResult | None = None
    total_results: int = 0
    unity_scale_count: int = 0
    precision_level: int = 2
    unity_scale_preference_active: bool = True
    precision_analysis: list[PrecisionAnalysisResult] | None = None

    def __post_init__(self):
        """Initialize precision_analysis if not provided."""
        if self.precision_analysis is None:
            self.precision_analysis = []


@dataclass(frozen=True)
class ThumbnailResult:
    """
    Complete thumbnail detection result.

    This is the main result object returned by the thumbnail finder,
    containing all the information needed to understand the detection.

    Attributes:
        confidence: Overall confidence score (0.0 to 1.0)
        scale_fg_to_thumb: Scale factor from FG to thumbnail size
        x_thumb_in_bg: X position of thumbnail in background
        y_thumb_in_bg: Y position of thumbnail in background
        scale_bg_to_fg: Scale factor from BG to FG size
        x_fg_in_scaled_bg: X position of FG in upscaled background
        y_fg_in_scaled_bg: Y position of FG in upscaled background
        analysis_data: Additional analysis information
        fg_size: Original foreground size (width, height)
        bg_size: Original background size (width, height)
        thumbnail_size: Calculated thumbnail size (width, height)
        upscaled_bg_size: Calculated upscaled background size (width, height)

    Used in:
    - vidkompy/align/__init__.py
    - vidkompy/align/core.py
    - vidkompy/align/display.py
    """

    confidence: float
    scale_fg_to_thumb: float
    x_thumb_in_bg: int
    y_thumb_in_bg: int
    scale_bg_to_fg: float
    x_fg_in_scaled_bg: int
    y_fg_in_scaled_bg: int
    analysis_data: AnalysisData
    fg_size: tuple[int, int] = (0, 0)
    bg_size: tuple[int, int] = (0, 0)
    thumbnail_size: tuple[int, int] = (0, 0)
    upscaled_bg_size: tuple[int, int] = (0, 0)

    @property
    def fg_width(self) -> int:
        """Get foreground width.

        Used in:
        - vidkompy/align/display.py
        """
        return self.fg_size[0]

    @property
    def fg_height(self) -> int:
        """Get foreground height.

        Used in:
        - vidkompy/align/display.py
        """
        return self.fg_size[1]

    @property
    def bg_width(self) -> int:
        """Get background width.

        Used in:
        - vidkompy/align/display.py
        """
        return self.bg_size[0]

    @property
    def bg_height(self) -> int:
        """Get background height.

        Used in:
        - vidkompy/align/display.py
        """
        return self.bg_size[1]

    @property
    def thumbnail_width(self) -> int:
        """Get thumbnail width.

        Used in:
        - vidkompy/align/display.py
        """
        return self.thumbnail_size[0]

    @property
    def thumbnail_height(self) -> int:
        """Get thumbnail height.

        Used in:
        - vidkompy/align/display.py
        """
        return self.thumbnail_size[1]


@dataclass
class FrameExtractionResult:
    """
    Result of frame extraction from video or image.

    Attributes:
        frames: List of extracted frames as numpy arrays
        original_size: Original size of the media (width, height)
        frame_count: Number of frames extracted
        is_video: Whether the source was a video file
        extraction_time: Time taken for extraction

    Used in:
    - vidkompy/align/__init__.py
    - vidkompy/align/frame_extractor.py
    """

    frames: list[np.ndarray]
    original_size: tuple[int, int]
    frame_count: int
    is_video: bool
    extraction_time: float = 0.0

    @property
    def width(self) -> int:
        """Get frame width."""
        return self.original_size[0]

    @property
    def height(self) -> int:
        """Get frame height."""
        return self.original_size[1]


# Type aliases for clarity
FrameArray = np.ndarray
ScaleRange = tuple[float, float]
Position = tuple[int, int]
Size = tuple[int, int]
