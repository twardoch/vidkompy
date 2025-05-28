#!/usr/bin/env python3
# this_file: src/vidkompy/comp/video_processor.py

"""
Core video processing functionality.

Handles video I/O, metadata extraction, and frame operations.
"""

import cv2
import ffmpeg
import numpy as np
from pathlib import Path
from loguru import logger
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)
from rich.console import Console

from vidkompy.com.data_types import VideoInfo

console = Console()


class VideoProcessor:
    """Handles comp video processing operations.

    This module provides the foundation for all video I/O operations.
    It abstracts away the complexity of video codecs and formats.

    Why separate video processing:
    - Isolates platform-specific code
    - Makes testing easier (can mock video I/O)
    - Single place for optimization
    - Handles codec compatibility issues

    Why both OpenCV and FFmpeg:
    - OpenCV: Frame-accurate reading, computer vision operations
    - FFmpeg: Audio handling, codec support, fast encoding

    Used in:
    - vidkompy/comp/alignment_engine.py
    - vidkompy/comp/temporal_sync.py
    """

    def get_video_info(self, video_path: str) -> VideoInfo:
        """Extract video metadata using ffprobe.

        Why ffprobe instead of OpenCV:
        - More reliable metadata extraction
        - Handles all video formats
        - Provides accurate duration/framerate
        - Detects audio streams properly

        Why we need this info:
        - FPS determines temporal alignment strategy
        - Resolution needed for spatial alignment
        - Duration for progress estimation
        - Audio presence for alignment method selection

        Args:
            video_path: Path to video file

        Returns:
            VideoInfo object with metadata

        Raises:
            ValueError: If video cannot be probed

        Used in:
        - vidkompy/comp/alignment_engine.py
        """
        logger.debug(f"Probing video: {video_path}")

        try:
            probe = ffmpeg.probe(video_path)

            # Find video stream
            video_stream = next(
                (s for s in probe["streams"] if s["codec_type"] == "video"), None
            )

            if not video_stream:
                msg = f"No video stream found in {video_path}"
                raise ValueError(msg)

            # Extract properties
            width = int(video_stream["width"])
            height = int(video_stream["height"])

            # Parse frame rate
            fps_str = video_stream.get("r_frame_rate", "0/1")
            if "/" in fps_str:
                num, den = map(int, fps_str.split("/"))
                fps = num / den if den != 0 else 0
            else:
                fps = float(fps_str)

            duration = float(probe["format"].get("duration", 0))

            # Calculate frame count
            frame_count = int(video_stream.get("nb_frames", 0))
            if frame_count == 0 and duration > 0 and fps > 0:
                frame_count = int(duration * fps)

            # Check audio
            audio_stream = next(
                (s for s in probe["streams"] if s["codec_type"] == "audio"), None
            )

            has_audio = audio_stream is not None
            audio_sample_rate = None
            audio_channels = None

            if audio_stream:
                audio_sample_rate = int(audio_stream.get("sample_rate", 0))
                audio_channels = int(audio_stream.get("channels", 0))

            info = VideoInfo(
                width=width,
                height=height,
                fps=fps,
                duration=duration,
                frame_count=frame_count,
                has_audio=has_audio,
                audio_sample_rate=audio_sample_rate,
                audio_channels=audio_channels,
                path=video_path,
            )

            logger.info(
                f"Video info for {Path(video_path).name}: "
                f"{width}x{height}, {fps:.2f} fps, {duration:.2f}s, "
                f"{frame_count} frames, audio: {'yes' if has_audio else 'no'}"
            )

            return info

        except Exception as e:
            logger.error(f"Failed to probe video {video_path}: {e}")
            raise

    def extract_frames(
        self, video_path: str, frame_indices: list[int], resize_factor: float = 1.0
    ) -> list[np.ndarray]:
        """Extract specific frames from video.

        Why selective frame extraction:
        - Loading full video would exhaust memory
        - We only need specific frames for matching
        - Random access is fast with modern codecs

        Why resize option:
        - Faster processing on smaller frames
        - SSIM works fine at lower resolution
        - Reduces memory usage significantly

        Why progress bar for large extractions:
        - Frame seeking can be slow on some codecs
        - Users need feedback for long operations
        - Helps identify performance issues

        Args:
            video_path: Path to video file
            frame_indices: List of frame indices to extract
            resize_factor: Factor to resize frames (for performance)

        Returns:
            List of frames as numpy arrays

        Used in:
        - vidkompy/comp/alignment_engine.py
        - vidkompy/comp/temporal_sync.py
        """
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return frames

        try:
            # Only show progress for large frame extractions
            if len(frame_indices) > 50:
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TimeRemainingColumn(),
                    console=console,
                    transient=True,
                ) as progress:
                    task = progress.add_task(
                        f"    Extracting {len(frame_indices)} frames...",
                        total=len(frame_indices),
                    )

                    for idx in frame_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()

                        if ret:
                            if resize_factor != 1.0:
                                height, width = frame.shape[:2]
                                new_width = int(width * resize_factor)
                                new_height = int(height * resize_factor)
                                frame = cv2.resize(frame, (new_width, new_height))
                            frames.append(frame)
                        else:
                            logger.warning(
                                f"Failed to read frame {idx} from {video_path}"
                            )

                        progress.update(task, advance=1)
            else:
                # No progress bar for small extractions
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()

                    if ret:
                        if resize_factor != 1.0:
                            height, width = frame.shape[:2]
                            new_width = int(width * resize_factor)
                            new_height = int(height * resize_factor)
                            frame = cv2.resize(frame, (new_width, new_height))
                        frames.append(frame)
                    else:
                        logger.warning(f"Failed to read frame {idx} from {video_path}")

        finally:
            cap.release()

        return frames

    def extract_frame_range(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        step: int = 1,
        resize_factor: float = 1.0,
    ) -> list[tuple[int, np.ndarray]]:
        """Extract a range of frames with their indices.

        Args:
            video_path: Path to video
            start_frame: Starting frame index
            end_frame: Ending frame index (exclusive)
            step: Frame step size
            resize_factor: Resize factor for frames

        Returns:
            List of (frame_index, frame) tuples

        """
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return frames

        try:
            for idx in range(start_frame, end_frame, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()

                if ret:
                    if resize_factor != 1.0:
                        height, width = frame.shape[:2]
                        new_width = int(width * resize_factor)
                        new_height = int(height * resize_factor)
                        frame = cv2.resize(frame, (new_width, new_height))
                    frames.append((idx, frame))
                else:
                    break

        finally:
            cap.release()

        return frames

    def create_video_writer(
        self, output_path: str, width: int, height: int, fps: float, codec: str = "mp4v"
    ) -> cv2.VideoWriter:
        """Create OpenCV video writer.

        Why H.264/mp4v codec:
        - Universal compatibility
        - Good compression ratio
        - Hardware acceleration available
        - Supports high resolutions

        Why we write silent video first:
        - OpenCV VideoWriter doesn't handle audio
        - Gives us perfect frame control
        - Audio added later with FFmpeg

        Args:
            output_path: Output video path
            width: Video width
            height: Video height
            fps: Frame rate
            codec: Video codec (default mp4v)

        Returns:
            VideoWriter object

        Used in:
        - vidkompy/comp/alignment_engine.py
        """
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            msg = f"Failed to create video writer for {output_path}"
            raise ValueError(msg)

        return writer

    def extract_all_frames(
        self,
        video_path: str,
        resize_factor: float = 1.0,
        crop: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray | None:
        """Extract all frames from video as a numpy array.

        This is used by the precise alignment engine which needs
        access to all frames for multi-resolution processing.

        Args:
            video_path: Path to video file
            resize_factor: Factor to resize frames (for performance)
            crop: Optional (x, y, width, height) tuple to crop frames

        Returns:
            Array of frames or None if extraction fails

        Used in:
        - vidkompy/comp/temporal_sync.py
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None

        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate resized dimensions
            if resize_factor != 1.0:
                new_width = int(width * resize_factor)
                new_height = int(height * resize_factor)
            else:
                new_width, new_height = width, height

            # Apply crop dimensions if specified
            if crop:
                crop_x, crop_y, crop_w, crop_h = crop
                # Scale crop coordinates by resize factor
                crop_x = int(crop_x * resize_factor)
                crop_y = int(crop_y * resize_factor)
                crop_w = int(crop_w * resize_factor)
                crop_h = int(crop_h * resize_factor)

                # Ensure crop is within bounds
                crop_x = max(0, min(crop_x, new_width - 1))
                crop_y = max(0, min(crop_y, new_height - 1))
                crop_w = min(crop_w, new_width - crop_x)
                crop_h = min(crop_h, new_height - crop_y)

                final_width = crop_w
                final_height = crop_h
            else:
                final_width = new_width
                final_height = new_height

            # Pre-allocate array for efficiency
            frames = np.zeros(
                (frame_count, final_height, final_width, 3), dtype=np.uint8
            )

            if crop:
                logger.info(
                    f"Extracting all {frame_count} frames at {new_width}x{new_height}, cropped to {final_width}x{final_height}"
                )
            else:
                logger.info(
                    f"Extracting all {frame_count} frames at {new_width}x{new_height}"
                )

            # Extract frames with progress bar
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"    Extracting {frame_count} frames...",
                    total=frame_count,
                )

                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if resize_factor != 1.0:
                        frame = cv2.resize(frame, (new_width, new_height))

                    # Apply cropping if specified
                    if crop:
                        frame = frame[
                            crop_y : crop_y + crop_h, crop_x : crop_x + crop_w
                        ]

                    if frame_idx < frame_count:
                        frames[frame_idx] = frame
                        frame_idx += 1
                        progress.update(task, advance=1)
                    else:
                        logger.warning(f"More frames than expected in {video_path}")
                        break

            # Trim array if we got fewer frames than expected
            if frame_idx < frame_count:
                logger.warning(f"Got {frame_idx} frames, expected {frame_count}")
                frames = frames[:frame_idx]

            logger.info(f"Extracted {len(frames)} frames from {Path(video_path).name}")
            return frames

        except Exception as e:
            logger.error(f"Failed to extract frames from {video_path}: {e}")
            return None
        finally:
            cap.release()
