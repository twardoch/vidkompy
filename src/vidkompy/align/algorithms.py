#!/usr/bin/env python3
# this_file: src/vidkompy/align/algorithms.py

"""
Core algorithms for thumbnail detection and matching.

This module contains the various algorithms used for template matching,
feature detection, histogram correlation, and sub-pixel refinement.

"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .result_types import MatchResult
from src.vidkompy.utils.correlation import histogram_correlation
from src.vidkompy.utils.image import ensure_gray, resize_frame

logger = logging.getLogger(__name__)

try:
    from skimage.registration import phase_cross_correlation

    PHASE_CORRELATION_AVAILABLE = True
except ImportError:
    PHASE_CORRELATION_AVAILABLE = False


@dataclass
class PerformanceStats:
    """Container for performance timing statistics."""

    template_matching_time: float = 0.0
    parallel_matching_time: float = 0.0
    ballpark_estimation_time: float = 0.0
    feature_matching_time: float = 0.0
    phase_correlation_time: float = 0.0
    hybrid_matching_time: float = 0.0
    sub_pixel_refinement_time: float = 0.0


class FeatureDetector(Enum):
    """Enumeration of available feature detectors."""

    AKAZE = "akaze"
    ORB = "orb"
    SIFT = "sift"
    NONE = "none"


class TemplateMatchingAlgorithm:
    """
    Advanced template matching algorithm with multi-scale and parallel processing.

    This is the core algorithm for finding the best position and scale
    of a foreground image within a background image, with sophisticated
    optimization techniques.

    Used in:
    - vidkompy/align/__init__.py
    - vidkompy/align/core.py
    - vidkompy/align/precision.py
    """

    def __init__(self, verbose: bool = False):
        """Initialize the template matching algorithm."""
        self.method = cv2.TM_CCOEFF_NORMED
        self.verbose = verbose
        self.performance_stats = PerformanceStats()

    def _match_at_scale(
        self,
        fg_frame: np.ndarray,
        bg_frame: np.ndarray,
        scale: float,
        method: str = "template",
    ) -> MatchResult:
        """
        Private worker method to perform template matching at a specific scale.

        This is used by both match_at_multiple_scales and parallel_multiscale_template_matching.

        Args:
            fg_frame: Foreground frame (template)
            bg_frame: Background frame (search area)
            scale: Scale factor to apply to foreground
            method: Method identifier for result tracking

        Returns:
            MatchResult with best match position and confidence

        """
        # Scale the foreground frame using utils function
        if scale != 1.0:
            scaled_fg = resize_frame(fg_frame, scale=scale)
        else:
            scaled_fg = fg_frame

        # Ensure template is smaller than or equal to search area
        # Allow equal dimensions for unity scale matching
        if (
            scaled_fg.shape[0] > bg_frame.shape[0]
            or scaled_fg.shape[1] > bg_frame.shape[1]
        ):
            return MatchResult(confidence=0.0, x=0, y=0, scale=scale, method=method)

        # Perform template matching
        # bg_frame is the image to search in, scaled_fg is the template to find
        result = cv2.matchTemplate(bg_frame, scaled_fg, self.method)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        return MatchResult(
            confidence=float(max_val),
            x=int(max_loc[0]),
            y=int(max_loc[1]),
            scale=scale,
            method=method,
        )

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

        Used in:
        - vidkompy/align/core.py
        """
        return self._match_at_scale(fg_frame, bg_frame, scale, method)

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
            result = self._match_at_scale(fg_frame, bg_frame, scale, method)
            results.append(result)

        return results

    def ballpark_scale_estimation(
        self,
        template: np.ndarray,
        image: np.ndarray,
        scale_range: tuple[float, float] = (0.3, 1.0),
    ) -> tuple[float, float]:
        """
        Ultra-fast ballpark scale estimation using histogram correlation.

        Returns:
            Tuple of (estimated_scale, confidence)

        Used in:
        - vidkompy/align/precision.py
        """
        start_time = time.time()

        # Convert to grayscale and downsample for speed
        template_gray = ensure_gray(template)
        image_gray = ensure_gray(image)

        # Downsample for ultra-fast processing
        template_small = cv2.resize(template_gray, (64, 64))

        # Compute template histogram
        template_hist = cv2.calcHist([template_small], [0], None, [64], [0, 256])
        template_hist = template_hist.flatten()

        # Test coarse scale range
        scales = np.linspace(scale_range[0], scale_range[1], 10)
        best_scale = 1.0
        best_corr = -1.0

        for scale in scales:
            # Estimate thumbnail size at this scale
            thumb_h = int(template_gray.shape[0] * scale)
            thumb_w = int(template_gray.shape[1] * scale)

            if thumb_h < 32 or thumb_w < 32:
                continue
            if thumb_h >= image_gray.shape[0] or thumb_w >= image_gray.shape[1]:
                continue

            # Create scaled template
            cv2.resize(template_small, (int(64 * scale), int(64 * scale)))

            # Compute histogram correlation with multiple regions
            max_region_corr = -1.0
            step = max(1, min(image_gray.shape[0] // 4, image_gray.shape[1] // 4))

            for y in range(0, image_gray.shape[0] - thumb_h, step):
                for x in range(0, image_gray.shape[1] - thumb_w, step):
                    # Extract region and compute histogram
                    region = image_gray[y : y + thumb_h, x : x + thumb_w]
                    region_small = cv2.resize(
                        region, (int(64 * scale), int(64 * scale))
                    )
                    region_hist = cv2.calcHist(
                        [region_small], [0], None, [64], [0, 256]
                    )
                    region_hist = region_hist.flatten()

                    # Compute correlation
                    corr = histogram_correlation(template_hist, region_hist)
                    max_region_corr = max(max_region_corr, corr)

            if max_region_corr > best_corr:
                best_corr = max_region_corr
                best_scale = scale

        processing_time = time.time() - start_time
        self.performance_stats.ballpark_estimation_time += processing_time

        if self.verbose:
            logger.debug(
                f"Ballpark estimation: scale={best_scale:.3f}, confidence={best_corr:.3f}"
            )

        return best_scale, best_corr

    def parallel_multiscale_template_matching(
        self,
        template: np.ndarray,
        image: np.ndarray,
        scale_range: tuple[float, float] = (0.1, 1.0),
        scale_steps: int = 50,
        max_workers: int | None = None,
    ) -> MatchResult | None:
        """
        Parallel multi-scale template matching for improved performance.

        Args:
            template: Input frame (template to find)
            image: Output frame (image to search in)
            scale_range: Min and max scale factors to test
            scale_steps: Number of scale steps to test
            max_workers: Number of parallel workers (None for auto)

        Returns:
            MatchResult or None if no good match found

        Used in:
        - vidkompy/align/precision.py
        """
        start_time = time.time()

        # Convert to grayscale for faster processing
        template_gray = ensure_gray(template)
        image_gray = ensure_gray(image)

        # Generate scale factors
        scales = np.linspace(scale_range[0], scale_range[1], scale_steps)

        def process_scale(scale):
            """Process a single scale factor."""
            # Use the extracted _match_at_scale method
            result = self._match_at_scale(
                template_gray, image_gray, scale, "parallel_template"
            )

            if result.confidence == 0.0:
                return None

            # Apply small bias toward unity scale (1.0) for similar confidence scores
            adjusted_val = result.confidence
            if abs(scale - 1.0) < 0.05:  # Within 5% of unity scale
                adjusted_val *= 1.02  # Small 2% bonus for near-unity scales

            return (scale, result.x, result.y, result.confidence, adjusted_val)

        if self.verbose:
            logger.debug(
                f"Testing {scale_steps} scales from {scale_range[0]:.2f} to {scale_range[1]:.2f} using parallel processing"
            )

        # Process scales in parallel
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_scale, scale) for scale in scales]
            for future in futures:
                result = future.result()
                if result is not None:
                    results.append(result)

        if not results:
            return None

        # Find best result (using adjusted values for selection)
        best_result = max(
            results, key=lambda x: x[4]
        )  # Sort by adjusted correlation value
        scale, x, y, correlation, _ = best_result

        processing_time = time.time() - start_time
        self.performance_stats.parallel_matching_time += processing_time

        logger.debug(
            f"Parallel template matching: scale={scale:.3f}, pos=({x},{y}), correlation={correlation:.3f}"
        )

        return MatchResult(
            scale=scale,
            x=x,
            y=y,
            confidence=correlation,
            method="template_parallel",
            processing_time=processing_time,
        )


class FeatureMatchingAlgorithm:
    """
    Advanced feature-based matching with multiple detector support.

    This algorithm supports AKAZE, ORB, and SIFT detectors and uses
    sophisticated matching techniques for robust transformation estimation.

    Used in:
    - vidkompy/align/__init__.py
    - vidkompy/align/precision.py
    """

    def __init__(self, max_features: int = 1000, verbose: bool = False):
        """
        Initialize the feature matching algorithm.

        Args:
            max_features: Maximum number of features to detect
            verbose: Enable verbose logging

        """
        self.verbose = verbose
        self.performance_stats = PerformanceStats()

        # Initialize feature detectors
        self._init_feature_detectors(max_features)

    def _init_feature_detectors(self, max_features: int):
        """Initialize various feature detectors for robust matching."""
        self.available_detectors = set()
        self.detectors = {}

        try:
            # AKAZE - Best balance of speed and accuracy
            self.detectors[FeatureDetector.AKAZE] = cv2.AKAZE_create()
            self.available_detectors.add(FeatureDetector.AKAZE)
        except AttributeError:
            pass

        try:
            # ORB - Fastest option
            self.detectors[FeatureDetector.ORB] = cv2.ORB_create(nfeatures=max_features)
            self.available_detectors.add(FeatureDetector.ORB)
        except AttributeError:
            pass

        try:
            # SIFT - Most accurate but slower
            self.detectors[FeatureDetector.SIFT] = cv2.SIFT_create()
            self.available_detectors.add(FeatureDetector.SIFT)
        except AttributeError:
            pass

        if self.verbose:
            available_names = [d.value for d in self.available_detectors]
            logger.debug(f"Available detectors: {available_names}")

    def enhanced_feature_matching(
        self, template: np.ndarray, image: np.ndarray, detector_type: str = "auto"
    ) -> MatchResult | None:
        """
        Enhanced feature-based matching with multiple detector options.

        Args:
            template: Foreground frame to match
            image: Background frame to search in
            detector_type: "auto", "akaze", "orb", or "sift"

        Returns:
            MatchResult or None if matching failed

        Used in:
        - vidkompy/align/precision.py
        """
        start_time = time.time()

        # Choose detector
        if detector_type == "auto":
            if FeatureDetector.AKAZE in self.available_detectors:
                detector = self.detectors[FeatureDetector.AKAZE]
                detector_name = "AKAZE"
            elif FeatureDetector.ORB in self.available_detectors:
                detector = self.detectors[FeatureDetector.ORB]
                detector_name = "ORB"
            elif FeatureDetector.SIFT in self.available_detectors:
                detector = self.detectors[FeatureDetector.SIFT]
                detector_name = "SIFT"
            else:
                logger.error("No feature detectors available")
                return None
        elif (
            detector_type == "akaze"
            and FeatureDetector.AKAZE in self.available_detectors
        ):
            detector = self.detectors[FeatureDetector.AKAZE]
            detector_name = "AKAZE"
        elif detector_type == "orb" and FeatureDetector.ORB in self.available_detectors:
            detector = self.detectors[FeatureDetector.ORB]
            detector_name = "ORB"
        elif (
            detector_type == "sift" and FeatureDetector.SIFT in self.available_detectors
        ):
            detector = self.detectors[FeatureDetector.SIFT]
            detector_name = "SIFT"
        else:
            logger.error(f"Detector {detector_type} not available")
            return None

        # Convert to grayscale
        template_gray = ensure_gray(template)
        image_gray = ensure_gray(image)

        # Detect features
        kp1, des1 = detector.detectAndCompute(template_gray, None)
        kp2, des2 = detector.detectAndCompute(image_gray, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            if self.verbose:
                logger.debug(f"{detector_name}: Insufficient features detected")
            return None

        # Match features
        if detector_name in ["ORB", "AKAZE"]:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:  # SIFT
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        try:
            matches = matcher.knnMatch(des1, des2, k=2)
        except cv2.error:
            if self.verbose:
                logger.debug(f"{detector_name}: Feature matching failed")
            return None

        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            if self.verbose:
                logger.debug(
                    f"{detector_name}: Too few good matches ({len(good_matches)})"
                )
            return None

        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        # Estimate transformation using RANSAC
        try:
            if len(good_matches) >= 8:
                # Use estimateAffinePartial2D for similarity transform
                M, mask = cv2.estimateAffinePartial2D(
                    src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0
                )
            else:
                # Fallback to homography
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    # Convert homography to similarity transform
                    M = H[:2, :]
                else:
                    M = None

        except cv2.error:
            if self.verbose:
                logger.debug(f"{detector_name}: Transformation estimation failed")
            return None

        if M is None:
            if self.verbose:
                logger.debug(f"{detector_name}: Could not estimate transformation")
            return None

        # Extract scale and translation
        if M.shape[0] == 2:  # Affine transform
            scale_x = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
            scale_y = np.sqrt(M[1, 0] ** 2 + M[1, 1] ** 2)
            scale = (scale_x + scale_y) / 2.0
            tx, ty = M[0, 2], M[1, 2]
        else:
            logger.warning(f"Unexpected transformation matrix shape: {M.shape}")
            return None

        # Calculate confidence based on inliers
        inliers = np.sum(mask) if mask is not None else len(good_matches)
        confidence = inliers / len(good_matches)

        processing_time = time.time() - start_time
        self.performance_stats.feature_matching_time += processing_time

        if self.verbose:
            logger.debug(
                f"{detector_name}: scale={scale:.3f}, pos=({tx:.1f},{ty:.1f}), conf={confidence:.3f}, inliers={inliers}/{len(good_matches)}"
            )

        return MatchResult(
            scale=scale,
            x=int(tx),
            y=int(ty),
            confidence=confidence,
            method=f"feature_{detector_name.lower()}",
            processing_time=processing_time,
        )

    # Alias for backward compatibility
    match = enhanced_feature_matching

    def _extract_transform_from_homography(
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


class SubPixelRefinementAlgorithm:
    """
    Sub-pixel refinement for precise position estimation.

    This algorithm refines the position estimate by searching
    in a small neighborhood around the initial estimate.

    Used in:
    - vidkompy/align/__init__.py
    - vidkompy/align/precision.py
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

        Used in:
        - vidkompy/align/precision.py
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
                        x=round(x_test),
                        y=round(y_test),
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


class PhaseCorrelationAlgorithm:
    """
    Phase correlation algorithm for fast and accurate translation detection.

    Uses FFT-based phase correlation for sub-pixel accurate position estimation.

    Used in:
    - vidkompy/align/__init__.py
    - vidkompy/align/precision.py
    """

    def __init__(self, verbose: bool = False):
        """Initialize the phase correlation algorithm."""
        self.verbose = verbose
        self.performance_stats = PerformanceStats()

    def phase_correlation_matching(
        self, template: np.ndarray, image: np.ndarray, scale_estimate: float = 1.0
    ) -> MatchResult | None:
        """
        Use phase correlation for fast and accurate translation detection.

        Args:
            template: Foreground frame to match
            image: Background frame to search in
            scale_estimate: Initial scale estimate for template resizing

        Returns:
            MatchResult or None if phase correlation not available or failed

        """
        if not PHASE_CORRELATION_AVAILABLE:
            if self.verbose:
                logger.warning("Phase correlation not available - install scikit-image")
            return None

        start_time = time.time()

        try:
            # Convert to grayscale
            if len(template.shape) == 3:
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = template

            if len(image.shape) == 3:
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image_gray = image

            # Resize template to estimated scale
            scaled_template = cv2.resize(
                template_gray, None, fx=scale_estimate, fy=scale_estimate
            )

            # Ensure template fits in image
            if (
                scaled_template.shape[0] >= image_gray.shape[0]
                or scaled_template.shape[1] >= image_gray.shape[1]
            ):
                return None

            # Pad template to same size as image for phase correlation
            padded_template = np.zeros_like(image_gray)
            padded_template[: scaled_template.shape[0], : scaled_template.shape[1]] = (
                scaled_template
            )

            # Apply phase correlation
            shift, error, phase_diff = phase_cross_correlation(
                image_gray.astype(np.float32),
                padded_template.astype(np.float32),
                upsample_factor=10,  # Sub-pixel accuracy
            )

            processing_time = time.time() - start_time
            self.performance_stats.phase_correlation_time += processing_time

            # Convert shift to position
            y_shift, x_shift = shift
            confidence = 1.0 - error  # Convert error to confidence

            if self.verbose:
                logger.debug(
                    f"Phase correlation: shift=({x_shift:.2f}, {y_shift:.2f}), confidence={confidence:.3f}"
                )

            return MatchResult(
                scale=scale_estimate,
                x=int(x_shift),
                y=int(y_shift),
                confidence=confidence,
                method="phase_correlation",
                processing_time=processing_time,
            )

        except Exception as e:
            if self.verbose:
                logger.warning(f"Phase correlation failed: {e}")
            return None


class HybridMatchingAlgorithm:
    """
    Hybrid approach combining multiple matching algorithms for optimal results.

    Uses a cascaded approach:
    1. Feature matching for initial estimate
    2. Parallel template matching for verification
    3. Phase correlation for refinement (if available)

    Used in:
    - vidkompy/align/__init__.py
    - vidkompy/align/precision.py
    """

    def __init__(self, verbose: bool = False):
        """Initialize the hybrid matching algorithm."""
        self.verbose = verbose
        self.template_matcher = TemplateMatchingAlgorithm(verbose)
        self.feature_matcher = FeatureMatchingAlgorithm(verbose=verbose)
        self.phase_matcher = PhaseCorrelationAlgorithm(verbose)

        self.performance_stats = PerformanceStats()

    def hybrid_matching(
        self, template: np.ndarray, image: np.ndarray, method: str = "auto"
    ) -> MatchResult | None:
        """
        Hybrid approach combining multiple matching algorithms for optimal results.

        Args:
            template: Foreground frame to match
            image: Background frame to search in
            method: "auto", "fast", or "accurate"

        Returns:
            Best MatchResult from all methods

        Used in:
        - vidkompy/align/precision.py
        """
        start_time = time.time()
        results = []

        # Step 1: Feature-based initial estimate
        if method in ["auto", "accurate"]:
            feature_result = self.feature_matcher.enhanced_feature_matching(
                template, image
            )
            if feature_result is not None:
                results.append(feature_result)
                if self.verbose:
                    logger.debug(f"Feature matching result: {feature_result}")

        # Step 2: Template matching (parallel for speed)
        if method in ["auto", "fast", "accurate"]:
            # Adjust scale range based on feature result if available
            if results and results[0].confidence > 0.5:
                # Narrow search around feature estimate
                estimated_scale = results[0].scale
                scale_range = (
                    max(0.1, estimated_scale - 0.2),
                    min(1.0, estimated_scale + 0.2),
                )
                scale_steps = 20  # Fewer steps since we have a good estimate
            else:
                # Full search
                scale_range = (0.1, 1.0)
                scale_steps = 30 if method == "fast" else 50

            template_result = (
                self.template_matcher.parallel_multiscale_template_matching(
                    template, image, scale_range=scale_range, scale_steps=scale_steps
                )
            )
            if template_result is not None:
                results.append(template_result)
                if self.verbose:
                    logger.debug(f"Template matching result: {template_result}")

        # Step 3: Phase correlation refinement (if available and we have an estimate)
        if PHASE_CORRELATION_AVAILABLE and results and method in ["auto", "accurate"]:
            # Use best scale estimate so far
            best_result = max(results, key=lambda r: r.confidence)
            phase_result = self.phase_matcher.phase_correlation_matching(
                template, image, scale_estimate=best_result.scale
            )
            if phase_result is not None:
                results.append(phase_result)
                if self.verbose:
                    logger.debug(f"Phase correlation result: {phase_result}")

        if not results:
            return None

        # Select best result based on confidence and method consistency
        if len(results) == 1:
            best_result = results[0]
        else:
            # Weight results by confidence and method reliability
            weighted_results = []
            for result in results:
                weight = result.confidence
                # Give slight preference to template matching for accuracy
                if result.method.startswith("template"):
                    weight *= 1.1
                # Phase correlation gets bonus for precision
                elif result.method == "phase_correlation":
                    weight *= 1.05
                weighted_results.append((result, weight))

            best_result = max(weighted_results, key=lambda x: x[1])[0]

        # Create final result with combined processing time
        total_time = time.time() - start_time
        final_result = MatchResult(
            scale=best_result.scale,
            x=best_result.x,
            y=best_result.y,
            confidence=best_result.confidence,
            method=f"hybrid_{len(results)}methods",
            processing_time=total_time,
        )

        if self.verbose:
            logger.debug(f"Hybrid matching final result: {final_result}")
        return final_result
