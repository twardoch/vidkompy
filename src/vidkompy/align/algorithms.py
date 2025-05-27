#!/usr/bin/env python3
# this_file: src/vidkompy/align/algorithms.py

"""
Core algorithms for thumbnail detection and matching.

This module contains the various algorithms used for template matching,
feature detection, histogram correlation, and sub-pixel refinement.
"""

import cv2
import numpy as np
from scipy.stats import pearsonr

from .result_types import MatchResult


class TemplateMatchingAlgorithm:
    """
    Template matching algorithm using OpenCV's matchTemplate.

    This is the comp algorithm for finding the best position and scale
    of a foreground image within a background image.
    """

    def __init__(self):
        """Initialize the template matching algorithm."""
        self.method = cv2.TM_CCOEFF_NORMED

    def match_template(
        self,
        fg_frame: np.ndarray,
        bg_frame: np.ndarray,
        scale: float = 1.0,
        method: str = "template",
    ) -> MatchResult:
        """
        Perform template matching at a specific scale.

        Args:
            fg_frame: Foreground frame (template)
            bg_frame: Background frame (search area)
            scale: Scale factor to apply to foreground
            method: Method identifier for result tracking

        Returns:
            MatchResult with best match position and confidence
        """
        # Scale the foreground frame
        if scale != 1.0:
            new_width = int(fg_frame.shape[1] * scale)
            new_height = int(fg_frame.shape[0] * scale)
            scaled_fg = cv2.resize(fg_frame, (new_width, new_height))
        else:
            scaled_fg = fg_frame

        # Ensure template is smaller than search area
        if (
            scaled_fg.shape[0] >= bg_frame.shape[0]
            or scaled_fg.shape[1] >= bg_frame.shape[1]
        ):
            return MatchResult(confidence=0.0, x=0, y=0, scale=scale, method=method)

        # Perform template matching
        result = cv2.matchTemplate(scaled_fg, bg_frame, self.method)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        return MatchResult(
            confidence=float(max_val),
            x=int(max_loc[0]),
            y=int(max_loc[1]),
            scale=scale,
            method=method,
        )

    def match_at_multiple_scales(
        self,
        fg_frame: np.ndarray,
        bg_frame: np.ndarray,
        scale_range: tuple[float, float] = (0.5, 1.5),
        scale_steps: int = 20,
        method: str = "multi_scale",
    ) -> list[MatchResult]:
        """
        Perform template matching at multiple scales.

        Args:
            fg_frame: Foreground frame (template)
            bg_frame: Background frame (search area)
            scale_range: Min and max scale factors
            scale_steps: Number of scale steps to test
            method: Method identifier for result tracking

        Returns:
            List of MatchResult objects for each scale tested
        """
        min_scale, max_scale = scale_range
        scales = np.linspace(min_scale, max_scale, scale_steps)
        results = []

        for scale in scales:
            result = self.match_template(fg_frame, bg_frame, scale, method)
            results.append(result)

        return results


class FeatureMatchingAlgorithm:
    """
    Feature-based matching using ORB detector and matcher.

    This algorithm detects keypoints and descriptors in both images
    and finds correspondences to estimate transformation parameters.
    """

    def __init__(self, max_features: int = 500):
        """
        Initialize the feature matching algorithm.

        Args:
            max_features: Maximum number of features to detect
        """
        self.orb = cv2.ORB_create(nfeatures=max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match_features(
        self, fg_frame: np.ndarray, bg_frame: np.ndarray, method: str = "feature"
    ) -> MatchResult | None:
        """
        Perform feature-based matching.

        Args:
            fg_frame: Foreground frame
            bg_frame: Background frame
            method: Method identifier for result tracking

        Returns:
            MatchResult if successful match found, None otherwise
        """
        # Convert to grayscale if needed
        fg_gray = self._ensure_grayscale(fg_frame)
        bg_gray = self._ensure_grayscale(bg_frame)

        # Detect keypoints and descriptors
        kp1, des1 = self.orb.detectAndCompute(fg_gray, None)
        kp2, des2 = self.orb.detectAndCompute(bg_gray, None)

        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return None

        # Match descriptors
        matches = self.matcher.match(des1, des2)

        if len(matches) < 4:
            return None

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography
        try:
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if homography is None:
                return None

            # Extract transformation parameters
            scale, x_offset, y_offset = self._extract_transform_params(homography)

            # Calculate confidence based on inlier ratio
            inliers = np.sum(mask)
            confidence = float(inliers) / len(matches)

            return MatchResult(
                confidence=confidence,
                x=int(x_offset),
                y=int(y_offset),
                scale=scale,
                method=method,
            )

        except cv2.error:
            return None

    def _ensure_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale if needed."""
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _extract_transform_params(
        self, homography: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Extract scale and translation from homography matrix.

        Args:
            homography: 3x3 homography matrix

        Returns:
            Tuple of (scale, x_offset, y_offset)
        """
        # Extract scale from the homography matrix
        scale_x = np.sqrt(homography[0, 0] ** 2 + homography[1, 0] ** 2)
        scale_y = np.sqrt(homography[0, 1] ** 2 + homography[1, 1] ** 2)
        scale = (scale_x + scale_y) / 2.0

        # Extract translation
        x_offset = homography[0, 2]
        y_offset = homography[1, 2]

        return scale, x_offset, y_offset


class HistogramCorrelationAlgorithm:
    """
    Fast histogram-based correlation for initial scale estimation.

    This algorithm provides a quick ballpark estimate of the scale
    by comparing color histograms of the images.
    """

    def __init__(self, bins: int = 64):
        """
        Initialize the histogram correlation algorithm.

        Args:
            bins: Number of histogram bins per channel
        """
        self.bins = bins

    def estimate_scale(
        self,
        fg_frame: np.ndarray,
        bg_frame: np.ndarray,
        scale_range: tuple[float, float] = (0.5, 1.5),
        scale_steps: int = 20,
    ) -> tuple[float, float]:
        """
        Estimate scale using histogram correlation.

        Args:
            fg_frame: Foreground frame
            bg_frame: Background frame
            scale_range: Min and max scale factors to test
            scale_steps: Number of scale steps

        Returns:
            Tuple of (best_scale, correlation_score)
        """
        # Calculate histogram of foreground
        fg_hist = self._calculate_histogram(fg_frame)

        min_scale, max_scale = scale_range
        scales = np.linspace(min_scale, max_scale, scale_steps)

        best_scale = 1.0
        best_correlation = 0.0

        for scale in scales:
            # Scale foreground and calculate histogram
            scaled_fg = self._scale_frame(fg_frame, scale)
            scaled_hist = self._calculate_histogram(scaled_fg)

            # Calculate correlation
            correlation = self._correlate_histograms(fg_hist, scaled_hist)

            if correlation > best_correlation:
                best_correlation = correlation
                best_scale = scale

        return best_scale, best_correlation

    def _calculate_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Calculate normalized histogram of frame."""
        if len(frame.shape) == 3:
            # Color image - calculate histogram for each channel
            hist_b = cv2.calcHist([frame], [0], None, [self.bins], [0, 256])
            hist_g = cv2.calcHist([frame], [1], None, [self.bins], [0, 256])
            hist_r = cv2.calcHist([frame], [2], None, [self.bins], [0, 256])
            hist = np.concatenate([hist_b, hist_g, hist_r])
        else:
            # Grayscale image
            hist = cv2.calcHist([frame], [0], None, [self.bins], [0, 256])

        # Normalize histogram
        hist = hist / (hist.sum() + 1e-10)
        return hist.flatten()

    def _scale_frame(self, frame: np.ndarray, scale: float) -> np.ndarray:
        """Scale frame by given factor."""
        new_width = int(frame.shape[1] * scale)
        new_height = int(frame.shape[0] * scale)
        return cv2.resize(frame, (new_width, new_height))

    def _correlate_histograms(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Calculate Pearson correlation between histograms."""
        try:
            correlation, _ = pearsonr(hist1, hist2)
            return abs(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0


class SubPixelRefinementAlgorithm:
    """
    Sub-pixel refinement for precise position estimation.

    This algorithm refines the position estimate by searching
    in a small neighborhood around the initial estimate.
    """

    def __init__(self, search_radius: int = 3):
        """
        Initialize the sub-pixel refinement algorithm.

        Args:
            search_radius: Radius of search neighborhood in pixels
        """
        self.search_radius = search_radius

    def refine_position(
        self, fg_frame: np.ndarray, bg_frame: np.ndarray, initial_result: MatchResult
    ) -> MatchResult:
        """
        Refine position estimate with sub-pixel accuracy.

        Args:
            fg_frame: Foreground frame
            bg_frame: Background frame
            initial_result: Initial match result to refine

        Returns:
            Refined MatchResult
        """
        # Create search grid around initial position
        x_center, y_center = initial_result.x, initial_result.y

        best_result = initial_result

        # Search in sub-pixel steps
        for dx in np.linspace(-self.search_radius, self.search_radius, 7):
            for dy in np.linspace(-self.search_radius, self.search_radius, 7):
                x_test = x_center + dx
                y_test = y_center + dy

                # Calculate correlation at this sub-pixel position
                correlation = self._calculate_subpixel_correlation(
                    fg_frame, bg_frame, x_test, y_test, initial_result.scale
                )

                if correlation > best_result.confidence:
                    best_result = MatchResult(
                        confidence=correlation,
                        x=int(round(x_test)),
                        y=int(round(y_test)),
                        scale=initial_result.scale,
                        method=f"{initial_result.method}_refined",
                    )

        return best_result

    def _calculate_subpixel_correlation(
        self,
        fg_frame: np.ndarray,
        bg_frame: np.ndarray,
        x: float,
        y: float,
        scale: float,
    ) -> float:
        """
        Calculate correlation at sub-pixel position.

        This is a simplified implementation - in practice, you might
        want to use more sophisticated interpolation methods.
        """
        try:
            # Scale foreground
            if scale != 1.0:
                new_width = int(fg_frame.shape[1] * scale)
                new_height = int(fg_frame.shape[0] * scale)
                scaled_fg = cv2.resize(fg_frame, (new_width, new_height))
            else:
                scaled_fg = fg_frame

            # Extract region from background at sub-pixel position
            x_int, y_int = int(x), int(y)

            # Check bounds
            if (
                x_int < 0
                or y_int < 0
                or x_int + scaled_fg.shape[1] >= bg_frame.shape[1]
                or y_int + scaled_fg.shape[0] >= bg_frame.shape[0]
            ):
                return 0.0

            bg_region = bg_frame[
                y_int : y_int + scaled_fg.shape[0], x_int : x_int + scaled_fg.shape[1]
            ]

            # Calculate normalized cross-correlation
            if bg_region.shape != scaled_fg.shape:
                return 0.0

            # Convert to float for correlation calculation
            fg_float = scaled_fg.astype(np.float32)
            bg_float = bg_region.astype(np.float32)

            # Normalize
            fg_norm = (fg_float - fg_float.mean()) / (fg_float.std() + 1e-10)
            bg_norm = (bg_float - bg_float.mean()) / (bg_float.std() + 1e-10)

            # Calculate correlation
            correlation = np.mean(fg_norm * bg_norm)

            return float(correlation)

        except:
            return 0.0
