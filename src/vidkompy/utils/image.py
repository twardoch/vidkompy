#!/usr/bin/env python3
# this_file: src/vidkompy/utils/image.py

"""
Image processing utilities for vidkompy.

This module provides common image processing functions used across
algorithms and extractors, focusing on grayscale conversion and
frame resizing operations.

Used in:
- vidkompy/align/algorithms.py
- vidkompy/align/frame_extractor.py
- vidkompy/comp/* (various modules)
"""

import cv2
import numpy as np

__all__ = ["ensure_gray", "resize_frame"]


def ensure_gray(frame: np.ndarray) -> np.ndarray:
    """
    Convert frame to grayscale if needed.

    Args:
        frame: Input frame (grayscale or color)

    Returns:
        Grayscale frame as 2D numpy array

    Note:
        If frame is already grayscale (2D), returns as-is.
        If frame is color (3D), converts using BGR2GRAY.

    Used in:
    - vidkompy/align/algorithms.py
    - vidkompy/align/core.py
    - vidkompy/align/frame_extractor.py
    - vidkompy/utils/__init__.py
    """
    if len(frame.shape) == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def resize_frame(
    frame: np.ndarray, scale: float | None = None, size: tuple[int, int] | None = None
) -> np.ndarray:
    """
    Resize frame by scale factor or to specific size.

    Args:
        frame: Input frame to resize
        scale: Scale factor (e.g., 0.5 for half size, 2.0 for double)
        size: Target size as (width, height) tuple

    Returns:
        Resized frame

    Raises:
        ValueError: If neither scale nor size is provided, or both are provided

    Note:
        If both scale and size are provided, size takes precedence.
        Uses INTER_AREA for downscaling, INTER_CUBIC for upscaling.

    Used in:
    - vidkompy/align/algorithms.py
    - vidkompy/align/frame_extractor.py
    - vidkompy/utils/__init__.py
    """
    if scale is None and size is None:
        msg = "Either scale or size must be provided"
        raise ValueError(msg)

    if size is not None:
        # Direct resize to specific size
        return cv2.resize(frame, size)

    if scale is not None:
        # Resize by scale factor
        if scale <= 0:
            msg = "Scale factor must be positive"
            raise ValueError(msg)

        height, width = frame.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Choose interpolation method based on scale
        if scale < 1.0:
            # Downscaling - use INTER_AREA for better quality
            interpolation = cv2.INTER_AREA
        else:
            # Upscaling - use INTER_CUBIC for smoother results
            interpolation = cv2.INTER_CUBIC

        return cv2.resize(frame, (new_width, new_height), interpolation=interpolation)

    # This shouldn't be reached, but just in case
    return frame
