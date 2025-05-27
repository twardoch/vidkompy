#!/usr/bin/env python3
# this_file: src/vidkompy/align/frame_extractor.py

"""
Frame extraction utilities for video and image processing.

This module handles extracting frames from video files and loading images,
with proper error handling and logging.
"""

import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from .result_types import FrameExtractionResult


class FrameExtractor:
    """
    Handles frame extraction from videos and image loading.

    This class provides methods to extract frames from video files
    and load images, returning standardized results with metadata.
    """

    def __init__(self):
        """Initialize the frame extractor."""

    def extract_frames(
        self, media_path: str | Path, max_frames: int = 7, verbose: bool = False
    ) -> FrameExtractionResult:
        """
        Extract frames from video or load image.

        Args:
            media_path: Path to video file or image
            max_frames: Maximum number of frames to extract
            verbose: Enable verbose logging

        Returns:
            FrameExtractionResult with extracted frames and metadata

        Raises:
            FileNotFoundError: If media file doesn't exist
            ValueError: If media file cannot be processed
        """
        start_time = time.time()
        media_path = Path(media_path)

        if not media_path.exists():
            msg = f"Media file not found: {media_path}"
            raise FileNotFoundError(msg)

        # Determine if it's a video or image based on extension
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        file_ext = media_path.suffix.lower()

        if file_ext in video_extensions:
            result = self._extract_frames_from_video(media_path, max_frames, verbose)
        elif file_ext in image_extensions:
            result = self._load_image_as_frames(media_path, verbose)
        else:
            msg = (
                f"Unsupported file format: {file_ext}. "
                f"Supported: {video_extensions | image_extensions}"
            )
            raise ValueError(msg)

        # Update extraction time
        extraction_time = time.time() - start_time
        return FrameExtractionResult(
            frames=result.frames,
            original_size=result.original_size,
            frame_count=result.frame_count,
            is_video=result.is_video,
            extraction_time=extraction_time,
        )

    def _extract_frames_from_video(
        self, video_path: Path, max_frames: int, verbose: bool
    ) -> FrameExtractionResult:
        """
        Extract frames from a video file.

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            verbose: Enable verbose logging

        Returns:
            FrameExtractionResult with video frames
        """
        if verbose:
            logger.info(f"Extracting frames from video: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            msg = f"Cannot open video file: {video_path}"
            raise ValueError(msg)

        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if total_frames == 0:
                msg = f"Video has no frames: {video_path}"
                raise ValueError(msg)

            # Calculate frame indices to extract
            if max_frames >= total_frames:
                frame_indices = list(range(total_frames))
            else:
                # Evenly distribute frames across the video
                step = total_frames / max_frames
                frame_indices = [int(i * step) for i in range(max_frames)]

            frames = []

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    logger.warning(f"Failed to read frame {frame_idx}")
                    continue

                frames.append(frame)

            if verbose:
                logger.info(f"Extracted {len(frames)} frames from video")

            return FrameExtractionResult(
                frames=frames,
                original_size=(width, height),
                frame_count=len(frames),
                is_video=True,
            )

        finally:
            cap.release()

    def _load_image_as_frames(
        self, image_path: Path, verbose: bool
    ) -> FrameExtractionResult:
        """
        Load an image as a single frame.

        Args:
            image_path: Path to image file
            verbose: Enable verbose logging

        Returns:
            FrameExtractionResult with single image frame
        """
        if verbose:
            logger.info(f"Loading image: {image_path}")

        image = cv2.imread(str(image_path))

        if image is None:
            msg = f"Cannot load image: {image_path}"
            raise ValueError(msg)

        height, width = image.shape[:2]

        if verbose:
            logger.info(f"Loaded image: {width}x{height}")

        return FrameExtractionResult(
            frames=[image], original_size=(width, height), frame_count=1, is_video=False
        )

    def preprocess_frame(
        self,
        frame: np.ndarray,
        target_size: tuple[int, int] | None = None,
        grayscale: bool = False,
    ) -> np.ndarray:
        """
        Preprocess a frame for analysis.

        Args:
            frame: Input frame array
            target_size: Optional target size (width, height)
            grayscale: Convert to grayscale if True

        Returns:
            Preprocessed frame
        """
        processed = frame.copy()

        # Convert to grayscale if requested
        if grayscale and len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # Resize if target size specified
        if target_size is not None:
            processed = cv2.resize(processed, target_size)

        return processed
