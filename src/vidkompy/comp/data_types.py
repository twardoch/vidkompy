#!/usr/bin/env python3
# this_file: src/vidkompy/comp/data_types.py

"""
Domain models for vidkompy composition module.

Contains all dataclasses and data structures used across the composition system.
"""

from dataclasses import dataclass
from pathlib import Path
from vidkompy.utils.enums import TimeMode


@dataclass
class AudioInfo:
    """Audio metadata container.

    Extracted from VideoInfo to reduce nullable fields.

    """

    sample_rate: int
    channels: int


@dataclass
class VideoInfo:
    """Video metadata container.

    Used in:
    - vidkompy/comp/align.py
    - vidkompy/comp/temporal.py
    - vidkompy/comp/video.py
    """

    width: int
    height: int
    fps: float
    duration: float
    frame_count: int
    has_audio: bool
    audio_info: AudioInfo | None = None
    path: str = ""

    @property
    def resolution(self) -> tuple[int, int]:
        """Get video resolution as (width, height)."""
        return (self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        return self.width / self.height if self.height > 0 else 0

    @classmethod
    def from_path(cls, video_path: Path | str) -> "VideoInfo":
        """Create VideoInfo by analyzing video file with ffprobe.

        Encapsulates ffprobe call currently scattered across VideoProcessor.

        Args:
            video_path: Path to video file

        Returns:
            VideoInfo instance with metadata

        """
        import subprocess
        import json

        video_path = Path(video_path)

        # Use ffprobe to get video metadata
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            # Find video and audio streams
            video_stream = None
            audio_stream = None

            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video" and video_stream is None:
                    video_stream = stream
                elif stream.get("codec_type") == "audio" and audio_stream is None:
                    audio_stream = stream

            if not video_stream:
                msg = f"No video stream found in {video_path}"
                raise ValueError(msg)

            # Extract video info
            width = int(video_stream["width"])
            height = int(video_stream["height"])
            fps = eval(video_stream["r_frame_rate"])  # e.g., "30/1" -> 30.0
            duration = float(video_stream.get("duration", 0))
            frame_count = int(video_stream.get("nb_frames", duration * fps))

            # Extract audio info if present
            audio_info = None
            has_audio = audio_stream is not None
            if has_audio:
                audio_info = AudioInfo(
                    sample_rate=int(audio_stream["sample_rate"]),
                    channels=int(audio_stream["channels"]),
                )

            return cls(
                width=width,
                height=height,
                fps=fps,
                duration=duration,
                frame_count=frame_count,
                has_audio=has_audio,
                audio_info=audio_info,
                path=str(video_path),
            )

        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            msg = f"Failed to analyze video {video_path}: {e}"
            raise RuntimeError(msg) from e


@dataclass
class FrameAlignment:
    """Represents alignment between a foreground and background frame.

    Used in:
    - vidkompy/comp/align.py
    - vidkompy/comp/dtw_aligner.py
    - vidkompy/comp/temporal.py
    - vidkompy/comp/tunnel.py
    """

    fg_frame_idx: int  # Foreground frame index (never changes)
    bg_frame_idx: int  # Corresponding background frame index
    similarity_score: float  # Similarity between frames (0-1)

    def __repr__(self) -> str:
        """"""
        return f"FrameAlignment(fg={self.fg_frame_idx}, bg={self.bg_frame_idx}, sim={self.similarity_score:.3f})"


@dataclass
class SpatialTransform:
    """Spatial transformation for overlaying foreground on background.

    Represents a 2D similarity transform (scaling + translation) for
    spatial alignment of videos.

    Used in:
    - vidkompy/comp/align.py
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

    def as_matrix(self) -> "np.ndarray":
        """Return 3x3 homography matrix for this spatial transform.

        Returns:
            3x3 transformation matrix for use with cv2.warpAffine

        """
        import numpy as np

        # 2D similarity transform: scale + translation
        matrix = np.array(
            [
                [self.scale_factor, 0, self.x_offset],
                [0, self.scale_factor, self.y_offset],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        return matrix


@dataclass
class TemporalSync:
    """Temporal alignment results.

    Used in:
    - vidkompy/comp/align.py
    - vidkompy/comp/temporal.py
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

    time_mode: TimeMode
    space_method: str
    skip_spatial: bool
    trim: bool
    max_keyframes: int = 2000
    verbose: bool = False
    border_thickness: int = 8
    blend: bool = False
    window: int = 0


__all__ = [
    "AudioInfo",
    "FrameAlignment",
    "ProcessingOptions",
    "SpatialTransform",
    "TemporalSync",
    "VideoInfo",
]
