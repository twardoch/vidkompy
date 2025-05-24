#!/usr/bin/env python3
# this_file: src/vidkompy/core/alignment_engine.py

"""
Main alignment engine that coordinates spatial and temporal alignment.

This is the high-level orchestrator that manages the complete video
overlay process.
"""

import tempfile
import cv2
import ffmpeg
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console

from ..models import (
    VideoInfo,
    MatchTimeMode,
    TemporalMethod,
    ProcessingOptions,
    SpatialAlignment,
    TemporalAlignment,
    FrameAlignment,
)
from .video_processor import VideoProcessor
from .spatial_alignment import SpatialAligner
from .temporal_alignment import TemporalAligner


console = Console()


class AlignmentEngine:
    """Orchestrates the complete video alignment and overlay process.

    This is the main coordinator that manages the entire video overlay workflow.
    It handles the high-level process flow while delegating specific tasks to
    specialized components.

    Why this architecture:
    - Separation of concerns: Each component (spatial, temporal, processing) has a single responsibility
    - Flexibility: Easy to swap alignment algorithms or add new methods
    - Testability: Each component can be tested independently
    - Progress tracking: Centralized progress reporting for better UX
    """

    def __init__(
        self,
        processor: VideoProcessor,
        verbose: bool = False,
        max_keyframes: int = 2000,
    ):
        """Initialize alignment engine.

        Args:
            processor: Video processor instance
            verbose: Enable verbose logging
            max_keyframes: Maximum keyframes for frame matching
        """
        self.processor = processor
        self.spatial_aligner = SpatialAligner()
        self.temporal_aligner = TemporalAligner(processor, max_keyframes)
        self.verbose = verbose

    def process(
        self,
        bg_path: str,
        fg_path: str,
        output_path: str,
        time_mode: MatchTimeMode,
        space_method: str,
        temporal_method: TemporalMethod,
        skip_spatial: bool,
        trim: bool,
    ):
        """Process video overlay with alignment.

        Args:
            bg_path: Background video path
            fg_path: Foreground video path
            output_path: Output video path
            time_mode: Temporal alignment mode
            space_method: Spatial alignment method
            temporal_method: Temporal algorithm to use (DTW or classic)
            skip_spatial: Skip spatial alignment
            trim: Trim to overlapping segment
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            # Analyze videos
            task = progress.add_task("Analyzing videos...", total=None)
            bg_info = self.processor.get_video_info(bg_path)
            fg_info = self.processor.get_video_info(fg_path)
            progress.update(task, completed=True)

            # Log compatibility
            self._log_compatibility(bg_info, fg_info)

            # Spatial alignment
            task = progress.add_task("Computing spatial alignment...", total=None)
            spatial_alignment = self._compute_spatial_alignment(
                bg_info, fg_info, space_method, skip_spatial
            )
            progress.update(task, completed=True)

            # Log spatial alignment results in non-verbose mode too
            if not skip_spatial:
                logger.info(
                    f"Spatial alignment result: offset=({spatial_alignment.x_offset}, {spatial_alignment.y_offset}), "
                    f"scale={spatial_alignment.scale_factor:.3f}, confidence={spatial_alignment.confidence:.3f}"
                )

            # Temporal alignment
            task = progress.add_task("Computing temporal alignment...", total=None)
            temporal_alignment = self._compute_temporal_alignment(
                bg_info, fg_info, time_mode, temporal_method, trim
            )
            progress.update(task, completed=True)

            # Log temporal alignment results
            logger.info(
                f"Temporal alignment result: method={temporal_alignment.method_used}, "
                f"offset={temporal_alignment.offset_seconds:.3f}s, "
                f"frames={len(temporal_alignment.frame_alignments)}, "
                f"confidence={temporal_alignment.confidence:.3f}"
            )

            # Compose final video
            task = progress.add_task("Composing output video...", total=None)
            self._compose_video(
                bg_info,
                fg_info,
                output_path,
                spatial_alignment,
                temporal_alignment,
                trim,
            )
            progress.update(task, completed=True)

            logger.info(f"✅ Processing complete: {output_path}")

    def _log_compatibility(self, bg_info: VideoInfo, fg_info: VideoInfo):
        """Log video compatibility information.

        Why we log compatibility:
        - Early warning of potential issues (e.g., fg larger than bg)
        - Helps users understand processing decisions
        - Aids in debugging alignment problems
        - Documents the input characteristics for reproducibility
        """
        logger.info("Video compatibility check:")
        logger.info(
            f"  Resolution: {bg_info.width}x{bg_info.height} vs {fg_info.width}x{fg_info.height}"
        )
        logger.info(f"  FPS: {bg_info.fps:.2f} vs {fg_info.fps:.2f}")
        logger.info(f"  Duration: {bg_info.duration:.2f}s vs {fg_info.duration:.2f}s")
        logger.info(
            f"  Audio: {'yes' if bg_info.has_audio else 'no'} vs {'yes' if fg_info.has_audio else 'no'}"
        )

        if fg_info.width > bg_info.width or fg_info.height > bg_info.height:
            logger.warning(
                "⚠️  Foreground is larger than background - will be scaled down"
            )

    def _compute_spatial_alignment(
        self, bg_info: VideoInfo, fg_info: VideoInfo, method: str, skip: bool
    ) -> SpatialAlignment:
        """Compute spatial alignment using sample frames.

        Why we use middle frames for alignment:
        - Middle frames typically have the main content fully visible
        - Avoids potential black frames or transitions at start/end
        - Single frame is usually sufficient for static camera shots
        - Fast computation while maintaining accuracy

        Why we support skipping:
        - Sometimes users know the alignment (e.g., already centered)
        - Useful for testing temporal alignment independently
        - Speeds up processing when spatial alignment isn't needed
        """
        if skip:
            logger.info("Skipping spatial alignment - centering foreground")
            x_offset = (bg_info.width - fg_info.width) // 2
            y_offset = (bg_info.height - fg_info.height) // 2
            return SpatialAlignment(x_offset, y_offset, 1.0, 1.0)

        # Extract sample frames for alignment
        bg_frames = self.processor.extract_frames(
            bg_info.path, [bg_info.frame_count // 2]
        )
        fg_frames = self.processor.extract_frames(
            fg_info.path, [fg_info.frame_count // 2]
        )

        if not bg_frames or not fg_frames:
            logger.error("Failed to extract frames for spatial alignment")
            x_offset = (bg_info.width - fg_info.width) // 2
            y_offset = (bg_info.height - fg_info.height) // 2
            return SpatialAlignment(x_offset, y_offset, 1.0, 0.0)

        return self.spatial_aligner.align(bg_frames[0], fg_frames[0], method, skip)

    def _compute_temporal_alignment(
        self,
        bg_info: VideoInfo,
        fg_info: VideoInfo,
        mode: MatchTimeMode,
        temporal_method: TemporalMethod,
        trim: bool,
    ) -> TemporalAlignment:
        """Compute temporal alignment based on mode.

        Why we have multiple modes:
        - PRECISE: Frame-based matching for maximum accuracy
        - FAST: Audio-based when available, falls back to frames

        Why audio can be faster:
        - Audio correlation is a 1D signal comparison
        - Works well when videos have identical audio tracks
        - Provides global offset quickly

        Why we always fall back to frames:
        - Not all videos have audio
        - Audio might be out of sync with video
        - Frame matching handles all cases
        """
        # Configure temporal aligner based on method
        self.temporal_aligner.use_dtw = temporal_method == TemporalMethod.DTW
        if mode == MatchTimeMode.PRECISE:
            # Always use frame-based alignment
            return self.temporal_aligner.align_frames(bg_info, fg_info, trim)

        elif mode == MatchTimeMode.FAST:
            # Try audio first if available
            if bg_info.has_audio and fg_info.has_audio:
                logger.info("Attempting audio-based temporal alignment")

                with tempfile.TemporaryDirectory() as tmpdir:
                    bg_audio = Path(tmpdir) / "bg_audio.wav"
                    fg_audio = Path(tmpdir) / "fg_audio.wav"

                    # Extract audio
                    if self.processor.extract_audio(
                        bg_info.path, str(bg_audio)
                    ) and self.processor.extract_audio(fg_info.path, str(fg_audio)):
                        offset = self.temporal_aligner.align_audio(
                            str(bg_audio), str(fg_audio)
                        )

                        # Create simple frame alignment based on audio offset
                        frame_alignments = self._create_audio_based_alignment(
                            bg_info, fg_info, offset, trim
                        )

                        return TemporalAlignment(
                            offset_seconds=offset,
                            frame_alignments=frame_alignments,
                            method_used="audio",
                            confidence=0.8,
                        )
                    else:
                        logger.warning(
                            "Audio extraction failed, falling back to frames"
                        )
            else:
                logger.info("No audio available, using frame-based alignment")

            # Fallback to frame-based
            return self.temporal_aligner.align_frames(bg_info, fg_info, trim)

    def _create_audio_based_alignment(
        self, bg_info: VideoInfo, fg_info: VideoInfo, offset_seconds: float, trim: bool
    ) -> list[FrameAlignment]:
        """Create frame alignments based on audio offset."""
        alignments = []

        # Calculate frame offset
        bg_frame_offset = int(offset_seconds * bg_info.fps)

        # Determine range
        if trim:
            start_fg = 0
            end_fg = min(
                fg_info.frame_count,
                int((bg_info.duration - offset_seconds) * fg_info.fps),
            )
        else:
            start_fg = 0
            end_fg = fg_info.frame_count

        # Create alignments
        fps_ratio = bg_info.fps / fg_info.fps if fg_info.fps > 0 else 1.0

        for fg_idx in range(start_fg, end_fg):
            bg_idx = int(fg_idx * fps_ratio + bg_frame_offset)
            bg_idx = max(0, min(bg_idx, bg_info.frame_count - 1))

            alignments.append(
                FrameAlignment(
                    fg_frame_idx=fg_idx,
                    bg_frame_idx=bg_idx,
                    similarity_score=0.7,  # Assumed good match from audio
                )
            )

        return alignments

    def _compose_video(
        self,
        bg_info: VideoInfo,
        fg_info: VideoInfo,
        output_path: str,
        spatial: SpatialAlignment,
        temporal: TemporalAlignment,
        trim: bool,
    ):
        """Compose the final output video.

        Why we use a two-step process (silent video + audio):
        - OpenCV doesn't handle audio, but provides frame-accurate control
        - FFmpeg handles audio well but can have frame accuracy issues
        - Combining both gives us the best of both worlds

        Why we prefer FG audio:
        - FG video is considered "better quality" per requirements
        - FG frames drive the output timing
        - Keeping FG audio maintains sync with FG visuals
        """
        logger.info(f"Composing video with {temporal.method_used} temporal alignment")

        # Use OpenCV for frame-accurate composition
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create silent video first
            temp_video = Path(tmpdir) / "temp_silent.mp4"

            self._compose_with_opencv(
                bg_info, fg_info, str(temp_video), spatial, temporal.frame_alignments
            )

            # Add audio
            self._add_audio_track(
                str(temp_video), output_path, bg_info, fg_info, temporal
            )

    def _compose_with_opencv(
        self,
        bg_info: VideoInfo,
        fg_info: VideoInfo,
        output_path: str,
        spatial: SpatialAlignment,
        alignments: list[FrameAlignment],
    ):
        """Compose video using OpenCV for frame accuracy."""
        # Create video writer
        writer = self.processor.create_video_writer(
            output_path,
            bg_info.width,
            bg_info.height,
            fg_info.fps,  # Use FG fps to preserve all FG frames
        )

        # Open video captures
        cap_bg = cv2.VideoCapture(bg_info.path)
        cap_fg = cv2.VideoCapture(fg_info.path)

        if not cap_bg.isOpened() or not cap_fg.isOpened():
            raise ValueError("Failed to open video files")

        try:
            frames_written = 0
            total_frames = len(alignments)

            for i, alignment in enumerate(alignments):
                # Read frames
                cap_fg.set(cv2.CAP_PROP_POS_FRAMES, alignment.fg_frame_idx)
                ret_fg, fg_frame = cap_fg.read()

                cap_bg.set(cv2.CAP_PROP_POS_FRAMES, alignment.bg_frame_idx)
                ret_bg, bg_frame = cap_bg.read()

                if not ret_fg or not ret_bg:
                    logger.warning(
                        f"Failed to read frames at fg={alignment.fg_frame_idx}, "
                        f"bg={alignment.bg_frame_idx}"
                    )
                    continue

                # Apply spatial alignment
                composite = self._overlay_frames(bg_frame, fg_frame, spatial)

                writer.write(composite)
                frames_written += 1

                # Progress update
                if frames_written % 100 == 0:
                    pct = (frames_written / total_frames) * 100
                    logger.info(f"Composition progress: {pct:.1f}%")

            logger.info(f"Wrote {frames_written} frames")

        finally:
            cap_bg.release()
            cap_fg.release()
            writer.release()

    def _overlay_frames(
        self, bg_frame: np.ndarray, fg_frame: np.ndarray, spatial: SpatialAlignment
    ) -> np.ndarray:
        """Overlay foreground on background with spatial alignment."""
        composite = bg_frame.copy()

        # Apply scaling if needed
        if spatial.scale_factor != 1.0:
            new_w = int(fg_frame.shape[1] * spatial.scale_factor)
            new_h = int(fg_frame.shape[0] * spatial.scale_factor)
            fg_frame = cv2.resize(
                fg_frame, (new_w, new_h), interpolation=cv2.INTER_AREA
            )

        fg_h, fg_w = fg_frame.shape[:2]
        bg_h, bg_w = bg_frame.shape[:2]

        # Calculate ROI with bounds checking
        x_start = max(0, spatial.x_offset)
        y_start = max(0, spatial.y_offset)
        x_end = min(bg_w, spatial.x_offset + fg_w)
        y_end = min(bg_h, spatial.y_offset + fg_h)

        # Calculate foreground crop if needed
        fg_x_start = max(0, -spatial.x_offset)
        fg_y_start = max(0, -spatial.y_offset)
        fg_x_end = fg_x_start + (x_end - x_start)
        fg_y_end = fg_y_start + (y_end - y_start)

        # Overlay
        if x_end > x_start and y_end > y_start:
            composite[y_start:y_end, x_start:x_end] = fg_frame[
                fg_y_start:fg_y_end, fg_x_start:fg_x_end
            ]

        return composite

    def _add_audio_track(
        self,
        video_path: str,
        output_path: str,
        bg_info: VideoInfo,
        fg_info: VideoInfo,
        temporal: TemporalAlignment,
    ):
        """Add audio track to the composed video."""
        # Prefer foreground audio as it's "better quality"
        if fg_info.has_audio:
            audio_source = fg_info.path
            audio_offset = 0.0  # FG audio is already aligned
            logger.info("Using foreground audio track")
        elif bg_info.has_audio:
            audio_source = bg_info.path
            audio_offset = -temporal.offset_seconds  # Compensate for alignment
            logger.info("Using background audio track")
        else:
            # No audio, just copy video
            logger.info("No audio tracks available")
            Path(video_path).rename(output_path)
            return

        # Merge audio with ffmpeg
        try:
            input_video = ffmpeg.input(video_path)

            if audio_offset != 0:
                input_audio = ffmpeg.input(audio_source, itsoffset=audio_offset)
            else:
                input_audio = ffmpeg.input(audio_source)

            stream = ffmpeg.output(
                input_video["v"],
                input_audio["a"],
                output_path,
                c="copy",
                acodec="aac",
                shortest=None,
            )

            ffmpeg.run(stream, overwrite_output=True, capture_stderr=True)

        except ffmpeg.Error as e:
            logger.error(f"Audio merge failed: {e.stderr.decode()}")
            # Fallback - save without audio
            Path(video_path).rename(output_path)
            logger.warning("Saved video without audio")
