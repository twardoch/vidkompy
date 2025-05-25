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
from loguru import logger
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.console import Console

from vidkompy.models import (
    VideoInfo,
    MatchTimeMode,
    TemporalMethod,
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
        border_thickness: int = 8,
        blend: bool = False,
        window: int = 0,
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
            border_thickness: Border thickness for border matching mode
            blend: Enable smooth blending at frame edges
            window: Sliding window size for frame matching
        """
        # Analyze videos - quick task, use simple logging
        logger.info("Analyzing videos...")
        bg_info = self.processor.get_video_info(bg_path)
        fg_info = self.processor.get_video_info(fg_path)

        # Log compatibility
        self._log_compatibility(bg_info, fg_info)

        # Spatial alignment - quick task, use simple logging
        logger.info("Computing spatial alignment...")
        spatial_alignment = self._compute_spatial_alignment(
            bg_info, fg_info, space_method, skip_spatial
        )

        # Log spatial alignment results in non-verbose mode too
        if not skip_spatial:
            logger.info(
                f"Spatial alignment result: offset=({spatial_alignment.x_offset}, {spatial_alignment.y_offset}), "
                f"scale={spatial_alignment.scale_factor:.3f}, confidence={spatial_alignment.confidence:.3f}"
            )

        # Temporal alignment - potentially time-intensive, use progress tracking
        logger.info("Computing temporal alignment...")
        temporal_alignment = self._compute_temporal_alignment(
            bg_info,
            fg_info,
            time_mode,
            temporal_method,
            trim,
            spatial_alignment,
            border_thickness,
            window,
        )

        # Log temporal alignment results
        logger.info(
            f"Temporal alignment result: method={temporal_alignment.method_used}, "
            f"offset={temporal_alignment.offset_seconds:.3f}s, "
            f"frames={len(temporal_alignment.frame_alignments)}, "
            f"confidence={temporal_alignment.confidence:.3f}"
        )

        # Compose final video - time-intensive, use progress tracking
        logger.info("Composing output video...")
        self._compose_video(
            bg_info,
            fg_info,
            output_path,
            spatial_alignment,
            temporal_alignment,
            trim,
            blend,
            border_thickness,
        )

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

        return self.spatial_aligner.align(
            bg_frames[0], fg_frames[0], method, skip_alignment=skip
        )

    def _compute_temporal_alignment(
        self,
        bg_info: VideoInfo,
        fg_info: VideoInfo,
        mode: MatchTimeMode,
        temporal_method: TemporalMethod,
        trim: bool,
        spatial_alignment: SpatialAlignment,
        border_thickness: int,
        window: int,
    ) -> TemporalAlignment:
        """Compute temporal alignment based on mode.

        Why we have multiple modes:
        - BORDER: Border-based matching focusing on frame edges (default)
        - PRECISE: Frame-based matching for maximum accuracy

        Why we always use frame-based methods:
        - Provides precise visual synchronization
        - Works with all videos regardless of audio
        - Handles all edge cases reliably
        """
        # Configure temporal aligner based on method and window
        self.temporal_aligner.use_dtw = temporal_method == TemporalMethod.DTW
        if window > 0:
            self.temporal_aligner.dtw_aligner.set_window(window)

        if mode == MatchTimeMode.BORDER:
            # Use border-based alignment with mask
            logger.info(
                f"Using border-based temporal alignment (border thickness: {border_thickness}px)"
            )
            border_mask = self.temporal_aligner.create_border_mask(
                spatial_alignment, fg_info, bg_info, border_thickness
            )
            return self.temporal_aligner.align_frames_with_mask(
                bg_info, fg_info, trim, border_mask
            )

        elif mode == MatchTimeMode.PRECISE:
            # Always use frame-based alignment
            return self.temporal_aligner.align_frames(bg_info, fg_info, trim)

        else:
            # For any other mode, use frame-based alignment
            return self.temporal_aligner.align_frames(bg_info, fg_info, trim)

    def _compose_video(
        self,
        bg_info: VideoInfo,
        fg_info: VideoInfo,
        output_path: str,
        spatial: SpatialAlignment,
        temporal: TemporalAlignment,
        trim: bool,
        blend: bool = False,
        border_thickness: int = 8,
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
                bg_info,
                fg_info,
                str(temp_video),
                spatial,
                temporal.frame_alignments,
                blend,
                border_thickness,
            )

            # Add audio
            self._add_audio_track(
                str(temp_video), output_path, bg_info, fg_info, temporal
            )

    def _frame_generator(self, video_path: str):
        """Yields frames from a video file sequentially.

        This generator provides sequential frame access which is much faster
        than random seeking. It's designed to eliminate the costly seek operations
        that slow down compositing.

        Args:
            video_path: Path to video file

        Yields:
            tuple: (frame_index, frame_array) for each frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            msg = f"Cannot open video file: {video_path}"
            raise OSError(msg)

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame_idx, frame
                frame_idx += 1
        finally:
            cap.release()

    def _compose_with_opencv(
        self,
        bg_info: VideoInfo,
        fg_info: VideoInfo,
        output_path: str,
        spatial: SpatialAlignment,
        alignments: list[FrameAlignment],
        blend: bool = False,
        border_thickness: int = 8,
    ):
        """Compose video using sequential reads for maximum performance.

        This optimized version eliminates random seeking by reading both video
        files sequentially. This provides a 10-100x speedup compared to the
        previous random-access approach.

        How it works:
        - Create generators that read each video file sequentially
        - Advance each generator to the frame we need
        - Since alignments are typically in ascending order, we mostly move forward
        - Eliminates costly seek operations that were the main bottleneck
        """
        if not alignments:
            logger.warning("No frame alignments provided. Cannot compose video.")
            return

        writer = self.processor.create_video_writer(
            output_path,
            bg_info.width,
            bg_info.height,
            fg_info.fps,  # Use FG fps to preserve all FG frames
        )

        bg_gen = self._frame_generator(bg_info.path)
        fg_gen = self._frame_generator(fg_info.path)

        current_bg_frame = None
        current_fg_frame = None
        current_bg_idx = -1
        current_fg_idx = -1

        frames_written = 0
        total_frames = len(alignments)

        # Create blend mask if needed
        blend_mask = None
        if blend:
            blend_mask = self.temporal_aligner.create_blend_mask(
                spatial, fg_info, bg_info, border_thickness
            )
            logger.info(f"Using blend mode with {border_thickness}px gradient")

        try:
            # Use proper progress bar for video composition
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total} frames)"),
                TimeRemainingColumn(),
                console=console,
                transient=False,
            ) as progress:
                task = progress.add_task("Composing frames", total=total_frames)

                for _i, alignment in enumerate(alignments):
                    needed_fg_idx = alignment.fg_frame_idx
                    needed_bg_idx = alignment.bg_frame_idx

                    # Advance foreground generator to the needed frame
                    while current_fg_idx < needed_fg_idx:
                        try:
                            current_fg_idx, current_fg_frame = next(fg_gen)
                        except StopIteration:
                            logger.error("Reached end of foreground video unexpectedly")
                            break

                    # Advance background generator to the needed frame
                    while current_bg_idx < needed_bg_idx:
                        try:
                            current_bg_idx, current_bg_frame = next(bg_gen)
                        except StopIteration:
                            logger.error("Reached end of background video unexpectedly")
                            break

                    if current_fg_frame is None or current_bg_frame is None:
                        logger.error("Frame generator did not yield a frame. Aborting.")
                        break

                    # We now have the correct pair of frames
                    composite = self._overlay_frames(
                        current_bg_frame, current_fg_frame, spatial, blend_mask
                    )
                    writer.write(composite)
                    frames_written += 1

                    # Update progress bar
                    progress.update(task, advance=1)

        except StopIteration:
            logger.warning("Reached end of a video stream unexpectedly.")
        finally:
            writer.release()
            logger.info(f"Wrote {frames_written} frames to {output_path}")

    def _overlay_frames(
        self,
        bg_frame: np.ndarray,
        fg_frame: np.ndarray,
        spatial: SpatialAlignment,
        blend_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Overlay foreground on background with spatial alignment and optional blending."""
        composite = bg_frame.copy()

        # Apply scaling if needed
        if spatial.scale_factor != 1.0:
            new_w = int(fg_frame.shape[1] * spatial.scale_factor)
            new_h = int(fg_frame.shape[0] * spatial.scale_factor)
            fg_frame = cv2.resize(
                fg_frame, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
            # Also scale blend mask if provided
            if blend_mask is not None:
                blend_mask = cv2.resize(
                    blend_mask, (new_w, new_h), interpolation=cv2.INTER_AREA
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
            fg_crop = fg_frame[fg_y_start:fg_y_end, fg_x_start:fg_x_end]
            bg_slice = composite[y_start:y_end, x_start:x_end]

            if blend_mask is not None:
                # Apply alpha blending using the blend mask
                mask_crop = blend_mask[fg_y_start:fg_y_end, fg_x_start:fg_x_end]

                # Ensure mask has same dimensions for broadcasting
                if len(fg_crop.shape) == 3:  # Color image
                    mask_crop = np.stack([mask_crop] * 3, axis=2)

                # Alpha blend: result = fg * alpha + bg * (1 - alpha)
                composite[y_start:y_end, x_start:x_end] = (
                    fg_crop.astype(np.float32) * mask_crop
                    + bg_slice.astype(np.float32) * (1.0 - mask_crop)
                ).astype(np.uint8)
            else:
                # Simple overlay (original behavior)
                composite[y_start:y_end, x_start:x_end] = fg_crop

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
