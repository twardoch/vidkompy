#!/usr/bin/env python3
# this_file: src/vidkompy/comp/domain_models.py

"""
Domain models for vidkompy composition module.

Contains all dataclasses and data structures used across the composition system.
"""

from dataclasses import dataclass
from .enums import MatchTimeMode


@dataclass
class VideoInfo:
    """Video metadata container.

    Used in:
    - vidkompy/comp/alignment_engine.py
    - vidkompy/comp/temporal_alignment.py
    - vidkompy/comp/video_processor.py
    """

    width: int
    height: int
    fps: float
    duration: float
    frame_count: int
    has_audio: bool
    audio_sample_rate: int | None = None
    audio_channels: int | None = None
    path: str = ""

    @property
    def resolution(self) -> tuple[int, int]:
        """Get video resolution as (width, height)."""
        return (self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        return self.width / self.height if self.height > 0 else 0


@dataclass
class FrameAlignment:
    """Represents alignment between a foreground and background frame.

    Used in:
    - vidkompy/comp/alignment_engine.py
    - vidkompy/comp/dtw_aligner.py
    - vidkompy/comp/temporal_alignment.py
    - vidkompy/comp/tunnel_aligner.py
    """

    fg_frame_idx: int  # Foreground frame index (never changes)
    bg_frame_idx: int  # Corresponding background frame index
    similarity_score: float  # Similarity between frames (0-1)

    def __repr__(self) -> str:
        """"""
        return f"FrameAlignment(fg={self.fg_frame_idx}, bg={self.bg_frame_idx}, sim={self.similarity_score:.3f})"


@dataclass
class SpatialAlignment:
    """Spatial offset for overlaying foreground on background.

    Used in:
    - vidkompy/comp/alignment_engine.py
    - vidkompy/comp/spatial_alignment.py
    """

    x_offset: int
    y_offset: int
    scale_factor: float = 1.0
    confidence: float = 1.0

    @property
    def offset(self) -> tuple[int, int]:
        """Get offset as tuple."""
        return (self.x_offset, self.y_offset)


@dataclass
class TemporalAlignment:
    """Temporal alignment results.

    Used in:
    - vidkompy/comp/alignment_engine.py
    - vidkompy/comp/temporal_alignment.py
    """

    offset_seconds: float  # Time offset in seconds
    frame_alignments: list[FrameAlignment]  # Frame-by-frame mapping
    method_used: str  # Method that produced this alignment
    confidence: float = 1.0

    @property
    def start_frame(self) -> int | None:
        """Get first aligned foreground frame."""
        return self.frame_alignments[0].fg_frame_idx if self.frame_alignments else None

    @property
    def end_frame(self) -> int | None:
        """Get last aligned foreground frame."""
        return self.frame_alignments[-1].fg_frame_idx if self.frame_alignments else None


@dataclass
class ProcessingOptions:
    """Options for video processing."""

    time_mode: MatchTimeMode
    space_method: str
    skip_spatial: bool
    trim: bool
    max_keyframes: int = 2000
    verbose: bool = False
    border_thickness: int = 8
    blend: bool = False
    window: int = 0


__all__ = [
    "FrameAlignment",
    "ProcessingOptions",
    "SpatialAlignment",
    "TemporalAlignment",
    "VideoInfo",
]
