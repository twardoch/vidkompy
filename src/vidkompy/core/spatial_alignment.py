#!/usr/bin/env python3
# this_file: src/vidkompy/core/spatial_alignment.py

"""
Spatial alignment module for finding optimal overlay positions.

Implements template matching and feature-based alignment methods.
"""

import cv2
import numpy as np
from loguru import logger

from vidkompy.models import SpatialAlignment


class SpatialAligner:
    """Handles spatial alignment of foreground on background frames.

    This module finds the optimal position to place the foreground video
    within the background video frame.

    Why spatial alignment is important:
    - Videos might be cropped differently
    - One video might be a subset/window of the other
    - Automatic alignment avoids manual positioning
    - Ensures content overlap for best visual result
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
            method: Alignment method ('precise'/'template' or 'fast'/'feature')
            skip_alignment: If True, just center the foreground

        Returns:
            SpatialAlignment with offset and scale
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

        # Perform alignment
        if method in ["precise", "template"]:
            return self._template_matching(bg_frame, fg_frame, scale_factor)
        elif method in ["fast", "feature"]:
            return self._feature_matching(bg_frame, fg_frame, scale_factor)
        else:
            logger.warning(f"Unknown spatial method {method}, using center alignment")
            x_offset = (bg_w - fg_w) // 2
            y_offset = (bg_h - fg_h) // 2
            return SpatialAlignment(
                x_offset=x_offset,
                y_offset=y_offset,
                scale_factor=scale_factor,
                confidence=0.5,
            )

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

    def _feature_matching(
        self, bg_frame: np.ndarray, fg_frame: np.ndarray, scale_factor: float
    ) -> SpatialAlignment:
        """Find alignment using feature matching (ORB).

        More robust to changes but potentially less precise.

        Why feature matching:
        - Works even with compression artifacts
        - Handles brightness/contrast changes
        - Can match videos with slight differences
        - Robust to noise and distortions

        Why ORB (Oriented FAST and Rotated BRIEF):
        - Patent-free alternative to SIFT/SURF
        - Very fast computation
        - Good balance of speed and accuracy
        - Rotation invariant

        Why homography:
        - Handles perspective changes
        - Can detect more complex transformations
        - RANSAC removes outlier matches

        Limitations:
        - Less precise than template matching for exact subsets
        - Requires textured content (fails on uniform areas)
        - May produce false matches needing validation
        """
        logger.debug("Using feature matching for spatial alignment")

        # Convert to grayscale
        bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
        fg_gray = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2GRAY)

        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=500)

        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(bg_gray, None)
        kp2, des2 = orb.detectAndCompute(fg_gray, None)

        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            logger.warning("Not enough features for matching, using center alignment")
            bg_h, bg_w = bg_frame.shape[:2]
            fg_h, fg_w = fg_frame.shape[:2]
            return SpatialAlignment(
                x_offset=(bg_w - fg_w) // 2,
                y_offset=(bg_h - fg_h) // 2,
                scale_factor=scale_factor,
                confidence=0.3,
            )

        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if len(matches) < 4:
            logger.warning("Not enough matches, using center alignment")
            bg_h, bg_w = bg_frame.shape[:2]
            fg_h, fg_w = fg_frame.shape[:2]
            return SpatialAlignment(
                x_offset=(bg_w - fg_w) // 2,
                y_offset=(bg_h - fg_h) // 2,
                scale_factor=scale_factor,
                confidence=0.3,
            )

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched points
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is None:
            logger.warning("Failed to find homography, using center alignment")
            bg_h, bg_w = bg_frame.shape[:2]
            fg_h, fg_w = fg_frame.shape[:2]
            return SpatialAlignment(
                x_offset=(bg_w - fg_w) // 2,
                y_offset=(bg_h - fg_h) // 2,
                scale_factor=scale_factor,
                confidence=0.3,
            )

        # Transform corner points
        h, w = fg_gray.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )
        dst = cv2.perspectiveTransform(pts, M)

        # Calculate offset from top-left corner
        x_offset = int(dst[0][0][0])
        y_offset = int(dst[0][0][1])

        # Calculate confidence based on inliers
        matches_mask = mask.ravel().tolist() if mask is not None else []
        inlier_ratio = sum(matches_mask) / len(matches_mask) if matches_mask else 0

        logger.info(
            f"Feature match found at ({x_offset}, {y_offset}) "
            f"with {sum(matches_mask)} inliers ({inlier_ratio:.2%})"
        )

        return SpatialAlignment(
            x_offset=x_offset,
            y_offset=y_offset,
            scale_factor=scale_factor,
            confidence=inlier_ratio,
        )
