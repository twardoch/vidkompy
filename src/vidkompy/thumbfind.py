#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["opencv-python", "numpy", "fire", "rich", "pathlib", "numba", "scikit-image"]
# ///
# this_file: thumbfind.py

"""
Thumbnail finder - detects scaled and translated thumbnails of foreground images within background images/videos.

This tool finds the scale and position transformation needed to match a foreground image/video
within a background image/video using multi-scale template matching and feature-based approaches.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numba import jit
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

try:
    from skimage.registration import phase_cross_correlation

    PHASE_CORRELATION_AVAILABLE = True
except ImportError:
    PHASE_CORRELATION_AVAILABLE = False


@dataclass
class MatchResult:
    """Container for match results with confidence scoring."""

    scale: float
    x: int
    y: int
    confidence: float
    method: str = "unknown"
    processing_time: float = 0.0

    @property
    def scale_percentage(self) -> float:
        return self.scale * 100

    def __str__(self) -> str:
        return f"Scale: {self.scale_percentage:.2f}%, Position: ({self.x}, {self.y}), Confidence: {self.confidence:.3f}"


console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ThumbnailFinder:
    """
    Advanced thumbnail finder with multiple detection algorithms.

    Combines template matching, feature-based detection, and phase correlation
    for robust and accurate thumbnail detection across various scenarios.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Verbose mode enabled")

        # Initialize feature detectors
        self._init_feature_detectors()

        # Performance tracking
        self.performance_stats = {
            "template_matching_time": 0.0,
            "feature_matching_time": 0.0,
            "phase_correlation_time": 0.0,
            "total_processing_time": 0.0,
        }

    def _init_feature_detectors(self):
        """Initialize various feature detectors for robust matching."""
        try:
            # AKAZE - Best balance of speed and accuracy
            self.akaze = cv2.AKAZE_create()
            self.has_akaze = True
        except AttributeError:
            self.has_akaze = False

        try:
            # ORB - Fastest option
            self.orb = cv2.ORB_create(nfeatures=1000)
            self.has_orb = True
        except AttributeError:
            self.has_orb = False

        try:
            # SIFT - Most accurate but slower
            self.sift = cv2.SIFT_create()
            self.has_sift = True
        except AttributeError:
            self.has_sift = False

        logger.debug(
            f"Available detectors: AKAZE={self.has_akaze}, ORB={self.has_orb}, SIFT={self.has_sift}"
        )

    def extract_frames(self, path: str | Path, max_frames: int = 7) -> list[np.ndarray]:
        """
        Extract frames from image or video file.

        Args:
            path: Path to image or video file
            max_frames: Maximum number of frames to extract for videos

        Returns:
            List of frames as numpy arrays
        """
        path = Path(path)
        if not path.exists():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

        # Check if it's an image or video
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}

        ext = path.suffix.lower()

        if ext in image_extensions:
            # Load single image
            img = cv2.imread(str(path))
            if img is None:
                msg = f"Could not load image: {path}"
                raise ValueError(msg)
            return [img]

        elif ext in video_extensions:
            # Extract frames from video
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                msg = f"Could not open video: {path}"
                raise ValueError(msg)

            frames = []

            # Extract the first n frames from the beginning of the video
            for _frame_idx in range(max_frames):
                ret, frame = cap.read()
                if not ret:
                    break  # End of video or error
                frames.append(frame)

            cap.release()

            if not frames:
                msg = f"Could not extract any frames from video: {path}"
                raise ValueError(msg)

            return frames
        else:
            msg = f"Unsupported file format: {ext}"
            raise ValueError(msg)

    @staticmethod
    @jit(nopython=True)
    def compute_normalized_correlation(
        template: np.ndarray, image: np.ndarray
    ) -> float:
        """Fast normalized cross-correlation computation using Numba."""
        template_mean = np.mean(template)
        image_mean = np.mean(image)

        numerator = np.sum((template - template_mean) * (image - image_mean))
        template_var = np.sum((template - template_mean) ** 2)
        image_var = np.sum((image - image_mean) ** 2)

        if template_var == 0 or image_var == 0:
            return 0.0

        return numerator / np.sqrt(template_var * image_var)

    @staticmethod
    @jit(nopython=True)
    def histogram_correlation(hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Fast histogram correlation for ballpark scale estimation."""
        # Normalize histograms
        h1 = hist1 / (np.sum(hist1) + 1e-7)
        h2 = hist2 / (np.sum(hist2) + 1e-7)

        # Compute correlation coefficient
        mean1 = np.mean(h1)
        mean2 = np.mean(h2)

        numerator = np.sum((h1 - mean1) * (h2 - mean2))
        denominator = np.sqrt(np.sum((h1 - mean1) ** 2) * np.sum((h2 - mean2) ** 2))

        if denominator == 0:
            return 0.0

        return numerator / denominator

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
        """
        # Convert to grayscale and downsample for speed
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template

        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

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
            scaled_template = cv2.resize(
                template_small, (int(64 * scale), int(64 * scale))
            )

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
                    corr = self.histogram_correlation(template_hist, region_hist)
                    max_region_corr = max(max_region_corr, corr)

            if max_region_corr > best_corr:
                best_corr = max_region_corr
                best_scale = scale

        if self.verbose:
            logger.debug(
                f"Ballpark estimation: scale={best_scale:.3f}, confidence={best_corr:.3f}"
            )

        return best_scale, best_corr

    def multiscale_template_matching(
        self,
        template: np.ndarray,
        image: np.ndarray,
        scale_range: tuple[float, float] = (0.1, 1.0),
        scale_steps: int = 50,
    ) -> tuple[float, int, int, float]:
        """
        Perform multi-scale template matching to find best scale and position.

        Args:
            template: Input frame (template to find)
            image: Output frame (image to search in)
            scale_range: Min and max scale factors to test
            scale_steps: Number of scale steps to test

        Returns:
            Tuple of (best_scale, best_x, best_y, best_correlation)
        """
        best_correlation = -1
        best_scale = 1.0
        best_loc = (0, 0)

        # Convert to grayscale for faster processing
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template

        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

        # Generate scale factors
        scales = np.linspace(scale_range[0], scale_range[1], scale_steps)

        if self.verbose:
            logger.debug(
                f"Testing {scale_steps} scales from {scale_range[0]:.2f} to {scale_range[1]:.2f}"
            )

        for scale in scales:
            # Resize template
            new_width = int(template_gray.shape[1] * scale)
            new_height = int(template_gray.shape[0] * scale)

            # Skip if template becomes too small or larger than image
            if new_width < 10 or new_height < 10:
                continue
            if new_height >= image_gray.shape[0] or new_width >= image_gray.shape[1]:
                continue

            resized_template = cv2.resize(template_gray, (new_width, new_height))

            # Apply template matching
            result = cv2.matchTemplate(
                image_gray, resized_template, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            # Apply small bias toward unity scale (1.0) for similar confidence scores
            adjusted_val = max_val
            if abs(scale - 1.0) < 0.05:  # Within 5% of unity scale
                adjusted_val *= 1.02  # Small 2% bonus for near-unity scales

            if adjusted_val > best_correlation:
                best_correlation = max_val  # Store original correlation, not adjusted
                best_scale = scale
                best_loc = max_loc

        return best_scale, best_loc[0], best_loc[1], best_correlation

    def feature_based_matching(
        self, template: np.ndarray, image: np.ndarray, min_match_count: int = 10
    ) -> tuple[float | None, int | None, int | None, float | None]:
        """
        Use ORB features for robust template matching as backup method.

        Returns:
            Tuple of (scale, x, y, confidence) or (None, None, None, None) if failed
        """
        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=1000)

        # Convert to grayscale
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template

        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(template_gray, None)
        kp2, des2 = orb.detectAndCompute(image_gray, None)

        if des1 is None or des2 is None:
            return None, None, None, None

        # FLANN matcher for efficient matching
        FLANN_INDEX_LSH = 6
        index_params = {
            "algorithm": FLANN_INDEX_LSH,
            "table_number": 6,
            "key_size": 12,
            "multi_probe_level": 1,
        }
        search_params = {"checks": 50}
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        try:
            matches = flann.knnMatch(des1, des2, k=2)
        except cv2.error:
            # Fallback to brute force matcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = [[m] for m in matches]  # Convert to same format as knnMatch

        # Filter good matches using Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            elif len(match_pair) == 1:
                good_matches.append(match_pair[0])

        if len(good_matches) >= min_match_count:
            # Extract matching points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )

            # Find homography
            try:
                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if homography is not None:
                    # Extract scale and translation
                    scale_x = np.linalg.norm(homography[0, :2])
                    scale_y = np.linalg.norm(homography[1, :2])
                    scale = (scale_x + scale_y) / 2.0

                    tx = homography[0, 2]
                    ty = homography[1, 2]

                    # Confidence based on inliers
                    inliers = np.sum(mask) if mask is not None else len(good_matches)
                    confidence = inliers / len(good_matches)

                    return scale, int(tx), int(ty), confidence

            except cv2.error:
                pass

        return None, None, None, None

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
            self.performance_stats["phase_correlation_time"] += processing_time

            # Convert shift to position
            y_shift, x_shift = shift
            confidence = 1.0 - error  # Convert error to confidence

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
            logger.warning(f"Phase correlation failed: {e}")
            return None

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
        """
        start_time = time.time()

        # Choose detector
        if detector_type == "auto":
            if self.has_akaze:
                detector = self.akaze
                detector_name = "AKAZE"
            elif self.has_orb:
                detector = self.orb
                detector_name = "ORB"
            elif self.has_sift:
                detector = self.sift
                detector_name = "SIFT"
            else:
                logger.error("No feature detectors available")
                return None
        elif detector_type == "akaze" and self.has_akaze:
            detector = self.akaze
            detector_name = "AKAZE"
        elif detector_type == "orb" and self.has_orb:
            detector = self.orb
            detector_name = "ORB"
        elif detector_type == "sift" and self.has_sift:
            detector = self.sift
            detector_name = "SIFT"
        else:
            logger.error(f"Detector {detector_type} not available")
            return None

        # Convert to grayscale
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template

        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

        # Detect features
        kp1, des1 = detector.detectAndCompute(template_gray, None)
        kp2, des2 = detector.detectAndCompute(image_gray, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
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
            logger.debug(f"{detector_name}: Too few good matches ({len(good_matches)})")
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
            logger.debug(f"{detector_name}: Transformation estimation failed")
            return None

        if M is None:
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
        self.performance_stats["feature_matching_time"] += processing_time

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
        """
        start_time = time.time()

        # Convert to grayscale for faster processing
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template

        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

        # Generate scale factors
        scales = np.linspace(scale_range[0], scale_range[1], scale_steps)

        def process_scale(scale):
            """Process a single scale factor."""
            # Resize template
            new_width = int(template_gray.shape[1] * scale)
            new_height = int(template_gray.shape[0] * scale)

            # Skip if template becomes too small or larger than image
            if new_width < 10 or new_height < 10:
                return None
            if new_height >= image_gray.shape[0] or new_width >= image_gray.shape[1]:
                return None

            resized_template = cv2.resize(template_gray, (new_width, new_height))

            # Apply template matching
            result = cv2.matchTemplate(
                image_gray, resized_template, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            return (scale, max_loc[0], max_loc[1], max_val)

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

        # Find best result
        best_result = max(results, key=lambda x: x[3])  # Sort by correlation value
        scale, x, y, correlation = best_result

        processing_time = time.time() - start_time
        self.performance_stats["template_matching_time"] += processing_time

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

    def precise_refinement(
        self,
        template: np.ndarray,
        image: np.ndarray,
        initial_scale: float,
        initial_x: int,
        initial_y: int,
        search_radius: int = 5,
        scale_steps: int = 21,
    ) -> tuple[float, int, int, float]:
        """
        Perform precise refinement around the initial estimate to minimize error.

        Args:
            template: Foreground frame to match
            image: Background frame to search in
            initial_scale: Initial scale estimate
            initial_x: Initial x position estimate
            initial_y: Initial y position estimate
            search_radius: Pixel radius to search around initial position
            scale_steps: Number of scale variations to test

        Returns:
            Tuple of (refined_scale, refined_x, refined_y, best_correlation)
        """
        logger.debug(
            f"Starting precise refinement around scale={initial_scale:.3f}, pos=({initial_x},{initial_y})"
        )

        # Convert to grayscale for faster processing
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template

        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

        best_correlation = -1
        best_scale = initial_scale
        best_x = initial_x
        best_y = initial_y

        # Define scale search range (±2% around initial estimate)
        scale_margin = 0.02
        scale_min = max(0.1, initial_scale - scale_margin)
        scale_max = min(1.0, initial_scale + scale_margin)
        scales = np.linspace(scale_min, scale_max, scale_steps)

        # Define position search range
        x_positions = range(
            max(0, initial_x - search_radius), initial_x + search_radius + 1
        )
        y_positions = range(
            max(0, initial_y - search_radius), initial_y + search_radius + 1
        )

        total_tests = len(scales) * len(x_positions) * len(y_positions)
        logger.debug(f"Testing {total_tests} combinations for precise refinement")

        with Progress() as progress:
            if self.verbose:
                task = progress.add_task("Precise refinement...", total=total_tests)

            for scale in scales:
                # Resize template
                new_width = int(template_gray.shape[1] * scale)
                new_height = int(template_gray.shape[0] * scale)

                # Skip if template becomes too small or too large
                if new_width < 10 or new_height < 10:
                    continue
                if (
                    new_height >= image_gray.shape[0]
                    or new_width >= image_gray.shape[1]
                ):
                    continue

                resized_template = cv2.resize(template_gray, (new_width, new_height))

                for x in x_positions:
                    for y in y_positions:
                        # Check bounds
                        if (
                            x + new_width >= image_gray.shape[1]
                            or y + new_height >= image_gray.shape[0]
                        ):
                            continue

                        # Extract the region from the image
                        region = image_gray[y : y + new_height, x : x + new_width]

                        # Ensure same size
                        if region.shape != resized_template.shape:
                            continue

                        # Compute normalized correlation directly
                        correlation = self.compute_normalized_correlation(
                            resized_template.astype(np.float32),
                            region.astype(np.float32),
                        )

                        if correlation > best_correlation:
                            best_correlation = correlation
                            best_scale = scale
                            best_x = x
                            best_y = y

                        if self.verbose:
                            progress.advance(task)

        improvement = (
            best_correlation
            - cv2.matchTemplate(
                image_gray,
                cv2.resize(
                    template_gray,
                    (
                        int(template_gray.shape[1] * initial_scale),
                        int(template_gray.shape[0] * initial_scale),
                    ),
                ),
                cv2.TM_CCOEFF_NORMED,
            ).max()
        )

        logger.debug(
            f"Precise refinement completed: scale={best_scale:.4f}, pos=({best_x},{best_y}), "
            f"correlation={best_correlation:.4f}, improvement={improvement:.4f}"
        )

        return best_scale, best_x, best_y, best_correlation

    def hybrid_matching(
        self, template: np.ndarray, image: np.ndarray, method: str = "auto"
    ) -> MatchResult | None:
        """
        Hybrid approach combining multiple matching algorithms for optimal results.

        Uses a cascaded approach:
        1. Feature matching for initial estimate
        2. Parallel template matching for verification
        3. Phase correlation for refinement (if available)

        Args:
            template: Foreground frame to match
            image: Background frame to search in
            method: "auto", "fast", or "accurate"

        Returns:
            Best MatchResult from all methods
        """
        start_time = time.time()
        results = []

        # Step 1: Feature-based initial estimate
        if method in ["auto", "accurate"]:
            feature_result = self.enhanced_feature_matching(template, image)
            if feature_result is not None:
                results.append(feature_result)
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

            template_result = self.parallel_multiscale_template_matching(
                template, image, scale_range=scale_range, scale_steps=scale_steps
            )
            if template_result is not None:
                results.append(template_result)
                logger.debug(f"Template matching result: {template_result}")

        # Step 3: Phase correlation refinement (if available and we have an estimate)
        if PHASE_CORRELATION_AVAILABLE and results and method in ["auto", "accurate"]:
            # Use best scale estimate so far
            best_result = max(results, key=lambda r: r.confidence)
            phase_result = self.phase_correlation_matching(
                template, image, scale_estimate=best_result.scale
            )
            if phase_result is not None:
                results.append(phase_result)
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

        logger.debug(f"Hybrid matching final result: {final_result}")
        return final_result

    def multi_precision_matching(
        self, template: np.ndarray, image: np.ndarray, precision_level: int = 2
    ) -> MatchResult | None:
        """
        Multi-precision matching with graduated refinement levels.

        Precision levels:
        0: Ultra-fast ballpark (histogram correlation) - ~1ms
        1: Coarse template matching with wide scale steps - ~10ms
        2: Balanced feature + template matching - ~25ms
        3: Fine feature + focused template matching - ~50ms
        4: Precise sub-pixel refinement - ~200ms

        Args:
            template: Foreground frame to match
            image: Background frame to search in
            precision_level: Precision level 0-4

        Returns:
            MatchResult with incremental improvements at each level
        """
        start_time = time.time()
        console.print(f"\n[cyan]● Precision Level {precision_level}[/cyan]")

        # Level 0: Ultra-fast ballpark estimation
        ballpark_scale, ballpark_confidence = self.ballpark_scale_estimation(
            template, image
        )
        console.print(
            f"  Level 0 (Ballpark): scale ≈ {ballpark_scale:.1%}, confidence = {ballpark_confidence:.3f}"
        )

        if precision_level == 0:
            return MatchResult(
                scale=ballpark_scale,
                x=0,  # Ballpark doesn't estimate position
                y=0,
                confidence=ballpark_confidence,
                method="ballpark",
                processing_time=time.time() - start_time,
            )

        # Level 1: Coarse template matching around ballpark estimate
        search_margin = 0.2  # ±20% around ballpark estimate
        coarse_range = (
            max(0.1, ballpark_scale - search_margin),
            min(1.0, ballpark_scale + search_margin),
        )

        coarse_result = self.parallel_multiscale_template_matching(
            template, image, scale_range=coarse_range, scale_steps=10
        )

        if coarse_result:
            console.print(
                f"  Level 1 (Coarse): scale = {coarse_result.scale:.1%}, pos = ({coarse_result.x}, {coarse_result.y}), confidence = {coarse_result.confidence:.3f}"
            )
        else:
            console.print("  Level 1 (Coarse): No match found")

        if precision_level == 1:
            return coarse_result or MatchResult(
                scale=ballpark_scale,
                x=0,
                y=0,
                confidence=ballpark_confidence,
                method="coarse",
                processing_time=time.time() - start_time,
            )

        # Level 2: Balanced matching (quick feature + moderate template)
        feature_result = self.enhanced_feature_matching(template, image)

        # Use best result so far for focused template search
        base_result = coarse_result or (
            feature_result
            if feature_result and feature_result.confidence > ballpark_confidence
            else None
        )

        if base_result:
            # Moderate template matching around the estimate
            balanced_range = (
                max(0.1, base_result.scale - 0.1),
                min(1.0, base_result.scale + 0.1),
            )
            balanced_result = self.parallel_multiscale_template_matching(
                template, image, scale_range=balanced_range, scale_steps=15
            )
        else:
            balanced_result = None

        # Choose best of feature and balanced template results
        candidates = [r for r in [feature_result, balanced_result] if r is not None]
        if candidates:
            level2_result = max(candidates, key=lambda r: r.confidence)
            console.print(
                f"  Level 2 (Balanced): scale = {level2_result.scale:.2%}, pos = ({level2_result.x}, {level2_result.y}), confidence = {level2_result.confidence:.3f}"
            )
        else:
            level2_result = base_result
            if level2_result:
                console.print(
                    f"  Level 2 (Balanced): Using coarse result - scale = {level2_result.scale:.2%}, confidence = {level2_result.confidence:.3f}"
                )

        if precision_level == 2:
            return level2_result or MatchResult(
                scale=ballpark_scale,
                x=0,
                y=0,
                confidence=ballpark_confidence,
                method="balanced",
                processing_time=time.time() - start_time,
            )

        # Level 3: Fine matching (enhanced feature + focused template)
        # Run enhanced feature matching with better parameters
        feature_result_fine = self.enhanced_feature_matching(
            template, image, detector_type="auto"
        )

        # Use best result so far for very focused template search
        base_result = (
            level2_result
            or coarse_result
            or (
                feature_result_fine
                if feature_result_fine
                and feature_result_fine.confidence > ballpark_confidence
                else None
            )
        )

        if base_result:
            # Fine template matching around the estimate
            fine_range = (
                max(0.1, base_result.scale - 0.05),
                min(1.0, base_result.scale + 0.05),
            )
            fine_result = self.parallel_multiscale_template_matching(
                template, image, scale_range=fine_range, scale_steps=20
            )
        else:
            fine_result = None

        # Choose best of feature and fine template results
        candidates = [r for r in [feature_result_fine, fine_result] if r is not None]
        if candidates:
            level3_result = max(candidates, key=lambda r: r.confidence)
            console.print(
                f"  Level 3 (Fine): scale = {level3_result.scale:.2%}, pos = ({level3_result.x}, {level3_result.y}), confidence = {level3_result.confidence:.3f}"
            )
        else:
            level3_result = base_result
            if level3_result:
                console.print(
                    f"  Level 3 (Fine): Using previous result - scale = {level3_result.scale:.2%}, confidence = {level3_result.confidence:.3f}"
                )

        if precision_level == 3:
            return level3_result or MatchResult(
                scale=ballpark_scale,
                x=0,
                y=0,
                confidence=ballpark_confidence,
                method="fine",
                processing_time=time.time() - start_time,
            )

        # Level 4: Precise sub-pixel refinement
        if level3_result and level3_result.confidence > 0.3:
            refined_scale, refined_x, refined_y, refined_confidence = (
                self.precise_refinement(
                    template,
                    image,
                    level3_result.scale,
                    level3_result.x,
                    level3_result.y,
                    search_radius=3,
                    scale_steps=41,  # Higher precision
                )
            )

            precise_result = MatchResult(
                scale=refined_scale,
                x=refined_x,
                y=refined_y,
                confidence=refined_confidence,
                method="precise",
                processing_time=time.time() - start_time,
            )
            console.print(
                f"  Level 4 (Precise): scale = {precise_result.scale:.3%}, pos = ({precise_result.x}, {precise_result.y}), confidence = {precise_result.confidence:.4f}"
            )
            return precise_result
        else:
            console.print(
                "  Level 4 (Precise): Skipped - insufficient confidence from level 3"
            )
            return level3_result or MatchResult(
                scale=ballpark_scale,
                x=0,
                y=0,
                confidence=ballpark_confidence,
                method="precise_fallback",
                processing_time=time.time() - start_time,
            )

    def _find_thumbnail_in_frames(
        self,
        fg_frames: list[np.ndarray],
        bg_frames: list[np.ndarray],
        precision: int = 2,
        unity_scale: bool = True,
    ) -> tuple[float, float, float, float, float, float, float, dict]:
        """
        Find the best thumbnail match across all frame combinations using advanced algorithms.

        Args:
            fg_frames: List of foreground frames to match
            bg_frames: List of background frames to search in
            precision: Precision level 0-4 (0=fastest ballpark, 4=most accurate)
            unity_scale: If True, bias toward 100% scale (shift-only) detection

        Returns:
            Tuple of (confidence_pct, scale_down_pct, x_shift, y_shift, scale_up_pct, x_shift_upscaled, y_shift_upscaled, analysis_data)
        """
        best_result = None
        all_results = []
        unity_scale_results = []  # Track 100% scale results separately
        processing_start = time.time()

        total_combinations = len(fg_frames) * len(bg_frames)

        with Progress() as progress:
            task = progress.add_task(
                "Searching for thumbnails...", total=total_combinations
            )

            for i, fg_frame in enumerate(fg_frames):
                for j, bg_frame in enumerate(bg_frames):
                    # Use precision-based matching - each level uses different algorithms
                    result = self.multi_precision_matching(
                        fg_frame, bg_frame, precision_level=precision
                    )

                    if result is not None:
                        all_results.append(result)

                        # Track unity scale results separately
                        if abs(result.scale - 1.0) < 0.05:  # Within 5% of unity scale
                            unity_scale_results.append(result)

                        # Track best result
                        if (
                            best_result is None
                            or result.confidence > best_result.confidence
                        ):
                            best_result = result

                        if self.verbose:
                            logger.debug(f"FG frame {i} vs BG frame {j}: {result}")

                    progress.advance(task)

        if not all_results:
            logger.warning("No matches found")
            return (0.0, 0.0, 0, 0, 100.0, 0, 0, {})

        # Apply unity scale preference if enabled
        unity_scale_result = None
        if unity_scale and unity_scale_results:
            # Find best unity scale result
            unity_scale_result = max(unity_scale_results, key=lambda r: r.confidence)

            # If unity scale result is competitive (within 10% confidence), prefer it
            confidence_threshold = 0.1
            if best_result is None or unity_scale_result.confidence >= (
                best_result.confidence - confidence_threshold
            ):
                logger.debug(
                    f"Preferring unity scale result: confidence={unity_scale_result.confidence:.3f} vs scaled={best_result.confidence:.3f}"
                )
                best_result = unity_scale_result

        # Average results for robustness (weight by confidence)
        if len(all_results) > 1:
            weights = np.array([r.confidence for r in all_results])
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)

                avg_scale = np.average([r.scale for r in all_results], weights=weights)
                avg_x = np.average([r.x for r in all_results], weights=weights)
                avg_y = np.average([r.y for r in all_results], weights=weights)

                # Use averaged values if they're close to the best result
                if abs(avg_scale - best_result.scale) < 0.1:
                    best_result.scale = avg_scale
                if abs(avg_x - best_result.x) < 10:
                    best_result.x = int(avg_x)
                if abs(avg_y - best_result.y) < 10:
                    best_result.y = int(avg_y)

        # Apply precise refinement if requested
        if precise and fg_frames and bg_frames and best_result:
            logger.info("Applying precise refinement...")
            # Use middle frame from each set for refinement
            mid_fg_idx = len(fg_frames) // 2
            mid_bg_idx = len(bg_frames) // 2

            refined_scale, refined_x, refined_y, refined_confidence = (
                self.precise_refinement(
                    fg_frames[mid_fg_idx],
                    bg_frames[mid_bg_idx],
                    best_result.scale,
                    best_result.x,
                    best_result.y,
                )
            )

            # Use refined values if they improved the result
            if refined_confidence > best_result.confidence:
                logger.info(
                    f"Precise refinement improved confidence from {best_result.confidence:.4f} to {refined_confidence:.4f}"
                )
                best_result.scale = refined_scale
                best_result.x = refined_x
                best_result.y = refined_y
                best_result.confidence = refined_confidence
            else:
                logger.info(
                    f"Precise refinement did not improve result (kept original confidence {best_result.confidence:.4f})"
                )

        # Calculate results from best_result
        confidence_pct = best_result.confidence * 100
        scale_down_pct = best_result.scale * 100
        x_shift = best_result.x
        y_shift = best_result.y

        # Calculate reverse transformation (upscale background to match foreground)
        scale_up_pct = (1.0 / best_result.scale) * 100 if best_result.scale > 0 else 100
        x_shift_upscaled = (
            int(-best_result.x / best_result.scale) if best_result.scale > 0 else 0
        )
        y_shift_upscaled = (
            int(-best_result.y / best_result.scale) if best_result.scale > 0 else 0
        )

        # Prepare analysis data for dual result reporting
        analysis_data = {
            "unity_scale_result": None,
            "scaled_result": None,
            "total_results": len(all_results),
            "unity_scale_count": len(unity_scale_results),
            "precision_level": precision,
            "unity_scale": unity_scale,
        }

        if unity_scale_result:
            analysis_data["unity_scale_result"] = {
                "confidence": unity_scale_result.confidence,
                "scale": unity_scale_result.scale,
                "x": unity_scale_result.x,
                "y": unity_scale_result.y,
                "method": unity_scale_result.method,
            }

        # Find best non-unity scale result for comparison
        non_unity_results = [r for r in all_results if abs(r.scale - 1.0) >= 0.05]
        if non_unity_results:
            scaled_result = max(non_unity_results, key=lambda r: r.confidence)
            analysis_data["scaled_result"] = {
                "confidence": scaled_result.confidence,
                "scale": scaled_result.scale,
                "x": scaled_result.x,
                "y": scaled_result.y,
                "method": scaled_result.method,
            }

        # Update performance stats
        total_processing_time = time.time() - processing_start
        self.performance_stats["total_processing_time"] = total_processing_time

        if self.verbose:
            logger.info(f"Performance stats: {self.performance_stats}")
            logger.info(
                f"Best method: {best_result.method}, Precision level: {precision}, Total combinations: {total_combinations}"
            )
            logger.info(
                f"Unity scale results: {len(unity_scale_results)}, Non-unity: {len(non_unity_results)}"
            )

        return (
            confidence_pct,
            scale_down_pct,
            x_shift,
            y_shift,
            scale_up_pct,
            x_shift_upscaled,
            y_shift_upscaled,
            analysis_data,
        )

    def find_thumbnail(
        self,
        fg: str | Path,
        bg: str | Path,
        num_frames: int = 7,
        verbose: bool = False,
        precision: int = 2,
        unity_scale: bool = True,
    ) -> None:
        """
        Main entry point for the thumbnail finder.

        Args:
            fg: Path to foreground image or video (the "input"/original frameset)
            bg: Path to background image or video (the "output"/frameset containing thumbnail)
            num_frames: Maximum number of frames to process for videos (default: 7)
            verbose: Enable verbose logging
            precision: Precision level 0-4 (0=ballpark ~1ms, 1=coarse ~10ms, 2=balanced ~25ms, 3=fine ~50ms, 4=precise ~200ms)
            unity_scale: If True, bias toward 100% scale (shift-only) detection
        """
        frames = num_frames
        if verbose:
            self.verbose = True
            logging.getLogger().setLevel(logging.DEBUG)

        try:
            console.print("[bold blue]Thumbnail Finder[/bold blue]")
            console.print(f"Foreground: {fg}")
            console.print(f"Background: {bg}")
            console.print(f"Max frames: {frames}")
            precision_names = [
                "Ballpark (~1ms)",
                "Coarse (~10ms)",
                "Balanced (~25ms)",
                "Fine (~50ms)",
                "Precise (~200ms)",
            ]
            console.print(
                f"[green]Precision Level: {precision}[/green] - {precision_names[precision]}"
            )
            console.print()

            # Extract frames
            console.print("[yellow]Extracting frames...[/yellow]")
            fg_frames = self.extract_frames(fg, frames)
            bg_frames = self.extract_frames(bg, frames)

            console.print(f"Extracted {len(fg_frames)} foreground frames")
            console.print(f"Extracted {len(bg_frames)} background frames")
            console.print()

            # Find thumbnail
            results = self._find_thumbnail_in_frames(
                fg_frames,
                bg_frames,
                precision=precision,
                unity_scale=unity_scale,
            )
            (
                confidence_pct,
                scale_down_pct,
                x_shift,
                y_shift,
                scale_up_pct,
                x_shift_upscaled,
                y_shift_upscaled,
                analysis_data,
            ) = results

            # Get filenames for display
            fg_filename = Path(fg).name
            bg_filename = Path(bg).name

            # Calculate pixel dimensions for display
            fg_height, fg_width = fg_frames[0].shape[:2]
            bg_height, bg_width = bg_frames[0].shape[:2]

            # Thumbnail dimensions (FG scaled down)
            thumbnail_width = int(fg_width * scale_down_pct / 100)
            thumbnail_height = int(fg_height * scale_down_pct / 100)

            # Upscaled BG dimensions (BG scaled up to match FG)
            upscaled_bg_width = int(bg_width * scale_up_pct / 100)
            upscaled_bg_height = int(bg_height * scale_up_pct / 100)

            # Display results
            table = Table(title="Thumbnail Detection Results")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="magenta")
            table.add_column("Unit", style="green")

            table.add_row("Confidence", f"{confidence_pct:.2f}", "%")
            table.add_row("FG file", f"{fg_filename}", "")
            table.add_row("FG original size", f"{fg_width}×{fg_height}", "px")
            table.add_row("BG file", f"{bg_filename}", "")
            table.add_row("BG original size", f"{bg_width}×{bg_height}", "px")
            table.add_row("Scale (FG → thumbnail)", f"{scale_down_pct:.2f}", "%")
            table.add_row(
                "Thumbnail size", f"{thumbnail_width}×{thumbnail_height}", "px"
            )
            table.add_row("X shift (thumbnail in BG)", f"{x_shift}", "px")
            table.add_row("Y shift (thumbnail in BG)", f"{y_shift}", "px")
            table.add_row("Scale (BG → FG size)", f"{scale_up_pct:.2f}", "%")
            table.add_row(
                "Upscaled BG size", f"{upscaled_bg_width}×{upscaled_bg_height}", "px"
            )
            table.add_row("X shift (FG on upscaled BG)", f"{x_shift_upscaled}", "px")
            table.add_row("Y shift (FG on upscaled BG)", f"{y_shift_upscaled}", "px")

            console.print(table)

            # Also print in simple format for easy parsing
            console.print("\n[bold]Summary:[/bold]")
            console.print(f"FG file: {fg_filename}")
            console.print(f"BG file: {bg_filename}")
            console.print(f"Confidence: {confidence_pct:.2f}%")
            console.print(f"FG size: {fg_width}×{fg_height} px")
            console.print(f"BG size: {bg_width}×{bg_height} px")
            console.print(
                f"Scale down: {scale_down_pct:.2f}% → {thumbnail_width}×{thumbnail_height} px"
            )
            console.print(f"Position: ({x_shift}, {y_shift}) px")
            console.print(
                f"Scale up: {scale_up_pct:.2f}% → {upscaled_bg_width}×{upscaled_bg_height} px"
            )
            console.print(
                f"Reverse position: ({x_shift_upscaled}, {y_shift_upscaled}) px"
            )

            # Display dual result analysis if available
            if analysis_data and (
                analysis_data.get("unity_scale_result")
                or analysis_data.get("scaled_result")
            ):
                console.print("\n[bold cyan]Alternative Analysis:[/bold cyan]")

                if analysis_data.get("unity_scale_result"):
                    unity = analysis_data["unity_scale_result"]
                    console.print(
                        f"Unity scale (100%) option: confidence={unity['confidence']:.3f}, position=({unity['x']}, {unity['y']})"
                    )

                if analysis_data.get("scaled_result"):
                    scaled = analysis_data["scaled_result"]
                    console.print(
                        f"Scaled option: confidence={scaled['confidence']:.3f}, scale={scaled['scale'] * 100:.2f}%, position=({scaled['x']}, {scaled['y']})"
                    )

                console.print(
                    f"Preference mode: {'Unity scale preferred' if unity_scale else 'Best confidence wins'}"
                )
                console.print(
                    f"Total results analyzed: {analysis_data['total_results']} ({analysis_data['unity_scale_count']} near 100% scale)"
                )

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            if verbose:
                raise


def find_thumbnail(
    fg: str | Path,
    bg: str | Path,
    frames: int = 7,
    verbose: bool = False,
    precision: int = 2,
    unity_scale: bool = True,
):
    """
    Main entry point for the thumbnail finder.

    Args:
        fg: Path to foreground image or video
        bg: Path to background image or video
        frames: Maximum number of frames to process
        verbose: Enable verbose logging
        precision: Precision level 0-4 (0=ballpark ~1ms, 1=coarse ~10ms, 2=balanced ~25ms, 3=fine ~50ms, 4=precise ~200ms)
        unity_scale: Bias toward 100% scale detection
    """
    thumbfind = ThumbnailFinder()
    thumbfind.find_thumbnail(fg, bg, frames, verbose, precision, unity_scale)
