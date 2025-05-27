#!/usr/bin/env python3
# this_file: src/vidkompy/comp/spatial_alignment.py

"""
DEPRECATED: Spatial alignment module for finding optimal overlay positions.

This module has been replaced by the vidkompy.align module which provides
better accuracy, robustness, and multi-scale detection capabilities.

Use vidkompy.align.ThumbnailFinder instead.
"""

import warnings
import cv2
import numpy as np
from loguru import logger

from vidkompy.comp.models import SpatialAlignment

warnings.warn(
    "spatial_alignment.py is deprecated. Use vidkompy.align.ThumbnailFinder instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)


class SpatialAligner:
    """Handles spatial alignment of foreground on background frames.

    This module finds the optimal position to place the foreground video
    within the background video frame.

    Why spatial alignment is important:
    - Videos might be cropped differently
    - One video might be a subset/window of the other
    - Automatic alignment avoids manual positioning
    - Ensures content overlap for best visual result

    Used in:
    - vidkompy/comp/alignment_engine.py
    - vidkompy/comp/temporal_alignment.py
    """

    def align(
        self,
        bg_frame: np.ndarray,
        fg_frame: np.ndarray,
        method: str = "precise",
        skip_alignment: bool = False,
    ) -> SpatialAlignment:
        """Find optimal position for foreground on background.

        Args:
            bg_frame: Background frame
            fg_frame: Foreground frame
            method: Alignment method (ignored, always uses template matching)
            skip_alignment: If True, just center the foreground

        Returns:
            SpatialAlignment with offset and scale

        Used in:
        - vidkompy/comp/alignment_engine.py
        - vidkompy/comp/temporal_alignment.py
        """
        bg_h, bg_w = bg_frame.shape[:2]
        fg_h, fg_w = fg_frame.shape[:2]

        # Check if scaling needed
        scale_factor = 1.0
        if fg_w > bg_w or fg_h > bg_h:
            scale_factor = min(bg_w / fg_w, bg_h / fg_h)
            logger.warning(
                f"Foreground larger than background, scaling by {scale_factor:.3f}"
            )

            # Scale foreground
            new_w = int(fg_w * scale_factor)
            new_h = int(fg_h * scale_factor)
            fg_frame = cv2.resize(
                fg_frame, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
            fg_h, fg_w = new_h, new_w

        if skip_alignment:
            # Center alignment
            x_offset = (bg_w - fg_w) // 2
            y_offset = (bg_h - fg_h) // 2
            return SpatialAlignment(
                x_offset=x_offset,
                y_offset=y_offset,
                scale_factor=scale_factor,
                confidence=1.0,
            )

        # Always use template matching
        return self._template_matching(bg_frame, fg_frame, scale_factor)

    def _template_matching(
        self, bg_frame: np.ndarray, fg_frame: np.ndarray, scale_factor: float
    ) -> SpatialAlignment:
        """Find best position using template matching.

        Uses normalized cross-correlation to find the best match.

        Why template matching:
        - Works perfectly when FG is an exact subset of BG
        - Very fast for exact matches
        - High confidence scores for good matches
        - OpenCV implementation is highly optimized

        Why we use grayscale:
        - 3x faster than color matching
        - More robust to color shifts
        - Structural alignment matters more than color

        Limitations:
        - Fails if videos have different compression/artifacts
        - Sensitive to brightness/contrast changes
        - Requires FG to be subset of BG

        """
        logger.debug("Using template matching for spatial alignment")

        # Convert to grayscale
        bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
        fg_gray = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2GRAY)

        # Apply template matching
        result = cv2.matchTemplate(bg_gray, fg_gray, cv2.TM_CCOEFF_NORMED)

        # Find best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Top-left corner of match
        x_offset, y_offset = max_loc

        logger.info(
            f"Template match found at ({x_offset}, {y_offset}) "
            f"with confidence {max_val:.3f}"
        )

        return SpatialAlignment(
            x_offset=x_offset,
            y_offset=y_offset,
            scale_factor=scale_factor,
            confidence=float(max_val),
        )
