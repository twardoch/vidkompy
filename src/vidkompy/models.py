#!/usr/bin/env python3
# this_file: src/vidkompy/models.py

"""
Data models and enums for vidkompy.

Contains all shared data structures used across the application.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class MatchTimeMode(Enum):
    """Temporal alignment modes."""
    FAST = "fast"      # Try audio first, fallback to frames
    PRECISE = "precise"  # Use frame-based matching


class SpatialMethod(Enum):
    """Spatial alignment methods."""
    TEMPLATE = "template"  # Precise template matching
    FEATURE = "feature"    # Fast feature-based matching
    CENTER = "center"      # Simple center alignment


@dataclass
class VideoInfo:
    """Video metadata container."""
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int
    has_audio: bool
    audio_sample_rate: Optional[int] = None
    audio_channels: Optional[int] = None
    path: str = ""
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get video resolution as (width, height)."""
        return (self.width, self.height)
    
    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        return self.width / self.height if self.height > 0 else 0


@dataclass
class FrameAlignment:
    """Represents alignment between a foreground and background frame."""
    fg_frame_idx: int      # Foreground frame index (never changes)
    bg_frame_idx: int      # Corresponding background frame index
    similarity_score: float  # Similarity between frames (0-1)
    
    def __repr__(self) -> str:
        return f"FrameAlignment(fg={self.fg_frame_idx}, bg={self.bg_frame_idx}, sim={self.similarity_score:.3f})"


@dataclass 
class SpatialAlignment:
    """Spatial offset for overlaying foreground on background."""
    x_offset: int
    y_offset: int
    scale_factor: float = 1.0
    confidence: float = 1.0
    
    @property
    def offset(self) -> Tuple[int, int]:
        """Get offset as tuple."""
        return (self.x_offset, self.y_offset)


@dataclass
class TemporalAlignment:
    """Temporal alignment results."""
    offset_seconds: float  # Time offset in seconds
    frame_alignments: List[FrameAlignment]  # Frame-by-frame mapping
    method_used: str  # Method that produced this alignment
    confidence: float = 1.0
    
    @property
    def start_frame(self) -> Optional[int]:
        """Get first aligned foreground frame."""
        return self.frame_alignments[0].fg_frame_idx if self.frame_alignments else None
    
    @property 
    def end_frame(self) -> Optional[int]:
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