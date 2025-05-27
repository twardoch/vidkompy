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

import cv2
import numpy as np
import fire
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import logging
from numba import jit
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import dataclass

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

            if max_val > best_correlation:
                best_correlation = max_val
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

    def find_thumbnail(
        self,
        fg_frames: list[np.ndarray],
        bg_frames: list[np.ndarray],
        precise: bool = False,
        method: str = "auto",
    ) -> tuple[float, float, float, float, float, float, float]:
        """
        Find the best thumbnail match across all frame combinations using advanced algorithms.

        Args:
            fg_frames: List of foreground frames to match
            bg_frames: List of background frames to search in
            precise: Enable precise refinement after initial estimation
            method: Matching method - "auto", "fast", "accurate", "template", "feature"

        Returns:
            Tuple of (confidence_pct, scale_down_pct, x_shift, y_shift, scale_up_pct, x_shift_upscaled, y_shift_upscaled)
        """
        best_result = None
        all_results = []
        processing_start = time.time()

        total_combinations = len(fg_frames) * len(bg_frames)

        with Progress() as progress:
            task = progress.add_task(
                "Searching for thumbnails...", total=total_combinations
            )

            for i, fg_frame in enumerate(fg_frames):
                for j, bg_frame in enumerate(bg_frames):
                    # Use the selected matching method
                    if method in ["auto", "fast", "accurate"]:
                        # Use hybrid approach
                        match_method = "fast" if method == "fast" else "auto"
                        result = self.hybrid_matching(
                            fg_frame, bg_frame, method=match_method
                        )
                    elif method == "template":
                        # Pure template matching
                        result = self.parallel_multiscale_template_matching(
                            fg_frame, bg_frame
                        )
                    elif method == "feature":
                        # Pure feature matching
                        result = self.enhanced_feature_matching(fg_frame, bg_frame)
                    else:
                        # Fallback to hybrid auto
                        result = self.hybrid_matching(fg_frame, bg_frame, method="auto")

                    if result is not None:
                        all_results.append(result)

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
            return (0.0, 0.0, 0, 0, 100.0, 0, 0)

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

        # Update performance stats
        total_processing_time = time.time() - processing_start
        self.performance_stats["total_processing_time"] = total_processing_time

        if self.verbose:
            logger.info(f"Performance stats: {self.performance_stats}")
            logger.info(
                f"Best method: {best_result.method}, Total combinations: {total_combinations}"
            )

        return (
            confidence_pct,
            scale_down_pct,
            x_shift,
            y_shift,
            scale_up_pct,
            x_shift_upscaled,
            y_shift_upscaled,
        )

    def find_thumbnail(
        self,
        fg: str | Path,
        bg: str | Path,
        frames: int = 7,
        verbose: bool = False,
        precise: bool = False,
        method: str = "auto",
    ) -> None:
        """
        Main entry point for the thumbnail finder.

        Args:
            fg: Path to foreground image or video (the "input"/original frameset)
            bg: Path to background image or video (the "output"/frameset containing thumbnail)
            frames: Maximum number of frames to process for videos (default: 7)
            verbose: Enable verbose logging
            precise: Enable precise refinement after initial estimation (slower but more accurate)
            method: Matching method - "auto" (hybrid), "fast" (speed optimized), "accurate" (quality optimized), "template" (template only), "feature" (feature only)
        """
        if verbose:
            self.verbose = True
            logging.getLogger().setLevel(logging.DEBUG)

        try:
            console.print("[bold blue]Thumbnail Finder[/bold blue]")
            console.print(f"Foreground: {fg}")
            console.print(f"Background: {bg}")
            console.print(f"Max frames: {frames}")
            if precise:
                console.print(
                    "[yellow]Precise mode enabled - may take longer but provides higher accuracy[/yellow]"
                )
            console.print(f"Method: {method.upper()}")
            console.print()

            # Extract frames
            console.print("[yellow]Extracting frames...[/yellow]")
            fg_frames = self.extract_frames(fg, frames)
            bg_frames = self.extract_frames(bg, frames)

            console.print(f"Extracted {len(fg_frames)} foreground frames")
            console.print(f"Extracted {len(bg_frames)} background frames")
            console.print()

            # Find thumbnail
            results = self.find_thumbnail(
                fg_frames, bg_frames, precise=precise, method=method
            )
            (
                confidence_pct,
                scale_down_pct,
                x_shift,
                y_shift,
                scale_up_pct,
                x_shift_upscaled,
                y_shift_upscaled,
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

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            if verbose:
                raise


def find_thumbnail(
    fg: str | Path,
    bg: str | Path,
    frames: int = 7,
    verbose: bool = False,
    precise: bool = False,
    method: str = "auto",
):
    """
    Main entry point for the thumbnail finder.
    """
    thumbfind = ThumbnailFinder()
    thumbfind.find_thumbnail(fg, bg, frames, verbose, precise, method)
