#!/usr/bin/env python3
# this_file: src/vidkompy/comp/frame_fingerprint.py

"""
Fast frame fingerprinting system using perceptual hashing.

This module provides ultra-fast frame comparison capabilities that are
100-1000x faster than SSIM while maintaining good accuracy for similar frames.
"""

import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from loguru import logger
import time

try:
    from vidkompy.core.numba_optimizations import (
        compute_hamming_distances_batch,
        compute_histogram_correlation,
        compute_weighted_similarity,
    )

    NUMBA_AVAILABLE = True
except ImportError:
    logger.warning("Numba optimizations not available for fingerprinting")
    NUMBA_AVAILABLE = False


class FrameFingerprinter:
    """Ultra-fast frame comparison using perceptual hashing.

    Why perceptual hashing:
    - 100-1000x faster than SSIM
    - Robust to compression artifacts and minor color/brightness changes
    - Compact representation (64 bits per frame)
    - Works well for finding similar frames

    Why multiple hash algorithms:
    - Different hashes capture different aspects of the image
    - Combining them improves robustness
    - Reduces false positives/negatives
    """

    def __init__(self, log_init: bool = True):
        """Initialize fingerprinter with multiple hash algorithms.

        Args:
            log_init: Whether to log initialization messages
        """
        self.hashers = {}
        self._init_hashers(log_init)

        # Cache for computed fingerprints
        self.fingerprint_cache: dict[str, dict[int, dict[str, np.ndarray]]] = {}

        # Flag to control numba usage
        self.use_numba = NUMBA_AVAILABLE

    def _init_hashers(self, log_init: bool = True):
        """Initialize available hash algorithms.

        Why these specific algorithms:
        - PHash: Frequency domain analysis, good for structure
        - AverageHash: Average color, good for brightness
        - ColorMomentHash: Color distribution, good for color changes
        - MarrHildrethHash: Edge detection, good for shapes
        """
        try:
            self.hashers["phash"] = cv2.img_hash.PHash_create()
            if log_init:
                logger.debug("✓ PHash initialized")
        except AttributeError:
            logger.warning("PHash not available")

        try:
            self.hashers["ahash"] = cv2.img_hash.AverageHash_create()
            if log_init:
                logger.debug("✓ AverageHash initialized")
        except AttributeError:
            logger.warning("AverageHash not available")

        try:
            self.hashers["dhash"] = cv2.img_hash.ColorMomentHash_create()
            if log_init:
                logger.debug("✓ ColorMomentHash initialized")
        except AttributeError:
            logger.warning("ColorMomentHash not available")

        try:
            self.hashers["mhash"] = cv2.img_hash.MarrHildrethHash_create()
            if log_init:
                logger.debug("✓ MarrHildrethHash initialized")
        except AttributeError:
            logger.warning("MarrHildrethHash not available")

        if not self.hashers:
            msg = (
                "No perceptual hash algorithms available. "
                "Please install opencv-contrib-python."
            )
            raise RuntimeError(msg)

        if log_init:
            logger.info(f"Initialized {len(self.hashers)} hash algorithms")

    def compute_fingerprint(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """Compute multi-algorithm fingerprint for a frame.

        Args:
            frame: Input frame as numpy array

        Returns:
            Dictionary of hash algorithm names to hash values

        Why multiple algorithms:
        - Redundancy reduces errors
        - Different algorithms catch different changes
        - Weighted combination improves accuracy
        """
        # Standardize frame size for consistent hashing
        std_size = (64, 64)
        std_frame = cv2.resize(frame, std_size, interpolation=cv2.INTER_AREA)

        # Convert to grayscale for most hashes
        if len(std_frame.shape) == 3:
            gray_frame = cv2.cvtColor(std_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = std_frame

        fingerprint = {}

        # Compute each available hash
        for name, hasher in self.hashers.items():
            try:
                if name in ["phash", "ahash", "mhash"]:
                    # These work on grayscale
                    hash_value = hasher.compute(gray_frame)
                else:
                    # ColorMomentHash needs color
                    hash_value = hasher.compute(std_frame)

                fingerprint[name] = hash_value
            except Exception as e:
                logger.warning(f"Failed to compute {name}: {e}")

        # Add color histogram as additional feature
        if len(frame.shape) == 3:
            fingerprint["histogram"] = self._compute_color_histogram(std_frame)

        return fingerprint

    def compute_masked_fingerprint(
        self, frame: np.ndarray, mask: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Compute fingerprint for masked region of frame.

        Args:
            frame: Input frame
            mask: Binary mask (1 = include, 0 = exclude)

        Returns:
            Fingerprint dictionary
        """
        # Apply mask to frame
        masked_frame = frame.copy()
        if len(frame.shape) == 3:
            # Apply to all channels
            for c in range(frame.shape[2]):
                masked_frame[:, :, c] = frame[:, :, c] * mask
        else:
            masked_frame = frame * mask

        # Crop to bounding box of mask to focus on relevant region
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            # Empty mask, return default fingerprint
            return self.compute_fingerprint(frame)

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        cropped = masked_frame[rmin : rmax + 1, cmin : cmax + 1]

        # Compute fingerprint on cropped region
        return self.compute_fingerprint(cropped)

    def _compute_color_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Compute color histogram for additional discrimination.

        Why color histogram:
        - Captures global color distribution
        - Complements structure-based hashes
        - Fast to compute and compare
        """
        # Compute histogram for each channel
        hist_b = cv2.calcHist([frame], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [32], [0, 256])

        # Concatenate and normalize
        hist = np.concatenate([hist_b, hist_g, hist_r])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)  # Normalize

        return hist.astype(np.float32)

    def compare_fingerprints(
        self, fp1: dict[str, np.ndarray], fp2: dict[str, np.ndarray]
    ) -> float:
        """Compare two fingerprints and return similarity score.

        Args:
            fp1: First fingerprint
            fp2: Second fingerprint

        Returns:
            Similarity score between 0 and 1

        Why weighted combination:
        - PHash is most reliable for video frames
        - Other hashes help disambiguate
        - Histogram adds color information
        """
        # Try numba optimization for histogram comparison if available
        if self.use_numba and "histogram" in fp1 and "histogram" in fp2:
            try:
                hist_score = compute_histogram_correlation(
                    fp1["histogram"], fp2["histogram"]
                )
            except Exception:
                # Fallback to OpenCV
                hist_score = cv2.compareHist(
                    fp1["histogram"], fp2["histogram"], cv2.HISTCMP_CORREL
                )
        elif "histogram" in fp1 and "histogram" in fp2:
            hist_score = cv2.compareHist(
                fp1["histogram"], fp2["histogram"], cv2.HISTCMP_CORREL
            )
        else:
            hist_score = 0.0

        hist_score = max(0, hist_score)  # Ensure non-negative

        # Collect hash distances
        hash_distances = []
        hash_names = []

        for name in fp1:
            if name not in fp2 or name == "histogram":
                continue

            # Ensure uint8 type for NORM_HAMMING
            h1 = (
                fp1[name].astype(np.uint8) if fp1[name].dtype != np.uint8 else fp1[name]
            )
            h2 = (
                fp2[name].astype(np.uint8) if fp2[name].dtype != np.uint8 else fp2[name]
            )
            distance = cv2.norm(h1, h2, cv2.NORM_HAMMING)

            # Normalize to 0-1 distance
            max_bits = h1.shape[0] * 8
            normalized_distance = distance / max_bits
            hash_distances.append(normalized_distance)
            hash_names.append(name)

        if not hash_distances and hist_score == 0:
            return 0.0

        # Define weights
        weight_map = {
            "phash": 0.4,  # Most reliable
            "ahash": 0.2,  # Good for brightness
            "dhash": 0.2,  # Good for color
            "mhash": 0.1,  # Good for edges
            "histogram": 0.1,  # Global color
        }

        # Use numba optimization if available
        if self.use_numba and hash_distances:
            try:
                # Prepare weights array
                weights = np.array([weight_map.get(name, 0.1) for name in hash_names])
                weights = np.append(weights, weight_map.get("histogram", 0.1))

                # Convert to numpy arrays
                hash_dist_array = np.array(hash_distances, dtype=np.float64)

                return compute_weighted_similarity(hash_dist_array, hist_score, weights)
            except Exception:
                # Fallback to standard implementation
                pass

        # Standard implementation
        total_weight = 0
        total_score = 0

        # Add hash similarities
        for i, name in enumerate(hash_names):
            weight = weight_map.get(name, 0.1)
            similarity = 1.0 - hash_distances[i]
            total_score += similarity * weight
            total_weight += weight

        # Add histogram score
        if hist_score > 0:
            weight = weight_map.get("histogram", 0.1)
            total_score += hist_score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def precompute_video_fingerprints(
        self,
        video_path: str,
        frame_indices: list[int],
        video_processor,
        resize_factor: float = 0.25,
    ) -> dict[int, dict[str, np.ndarray]]:
        """Precompute fingerprints for all specified frames in parallel.

        Args:
            video_path: Path to video file
            frame_indices: List of frame indices to process
            video_processor: VideoProcessor instance for frame extraction
            resize_factor: Factor to resize frames before hashing

        Returns:
            Dictionary mapping frame indices to fingerprints

        Why parallel processing:
        - Frame extraction is I/O bound (use threads)
        - Hash computation is CPU bound (use processes)
        - Significant speedup on multi-comp systems
        """
        # Check cache first
        if video_path in self.fingerprint_cache:
            cached = self.fingerprint_cache[video_path]
            missing = [idx for idx in frame_indices if idx not in cached]
            if not missing:
                return {idx: cached[idx] for idx in frame_indices}
            frame_indices = missing

        logger.info(f"Computing fingerprints for {len(frame_indices)} frames...")
        start_time = time.time()

        # Step 1: Extract frames in batches (I/O bound)
        frames_dict = {}
        batch_size = 50

        for i in range(0, len(frame_indices), batch_size):
            batch_indices = frame_indices[i : i + batch_size]
            batch_frames = video_processor.extract_frames(
                video_path, batch_indices, resize_factor
            )

            for idx, frame in zip(batch_indices, batch_frames, strict=False):
                if frame is not None:
                    frames_dict[idx] = frame

        # Step 2: Compute fingerprints in parallel (CPU bound)
        fingerprints = {}

        # Use process pool for CPU-intensive hash computation
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self._compute_fingerprint_worker, frame): idx
                for idx, frame in frames_dict.items()
            }

            # Collect results
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    fingerprint = future.result()
                    fingerprints[idx] = fingerprint
                except Exception as e:
                    logger.warning(
                        f"Failed to compute fingerprint for frame {idx}: {e}"
                    )

        # Update cache
        if video_path not in self.fingerprint_cache:
            self.fingerprint_cache[video_path] = {}
        self.fingerprint_cache[video_path].update(fingerprints)

        elapsed = time.time() - start_time
        fps = len(fingerprints) / elapsed if elapsed > 0 else 0
        logger.info(
            f"Computed {len(fingerprints)} fingerprints in {elapsed:.2f}s "
            f"({fps:.1f} fps)"
        )

        return fingerprints

    @staticmethod
    def _compute_fingerprint_worker(frame: np.ndarray) -> dict[str, np.ndarray]:
        """Worker function for parallel fingerprint computation.

        Static method to enable pickling for multiprocessing.
        """
        # Create a new instance in the worker process without logging
        fingerprinter = FrameFingerprinter(log_init=False)
        return fingerprinter.compute_fingerprint(frame)

    def compare_fingerprints_batch(
        self, fps1: list[dict[str, np.ndarray]], fps2: list[dict[str, np.ndarray]]
    ) -> np.ndarray:
        """Batch comparison of fingerprints using numba optimization.

        Args:
            fps1: List of fingerprints from first set
            fps2: List of fingerprints from second set

        Returns:
            Similarity matrix (len(fps1), len(fps2))
        """
        n1, n2 = len(fps1), len(fps2)
        similarities = np.zeros((n1, n2), dtype=np.float32)

        if self.use_numba and n1 > 5 and n2 > 5:
            try:
                # Extract hash types from first fingerprint
                hash_types = [k for k in fps1[0].keys() if k != "histogram"]

                # Prepare batch arrays for each hash type
                for hash_type in hash_types:
                    if all(hash_type in fp for fp in fps1 + fps2):
                        # Extract hashes for this type
                        hashes1 = np.array([fp[hash_type].flatten() for fp in fps1])
                        hashes2 = np.array([fp[hash_type].flatten() for fp in fps2])

                        # Compute batch distances
                        distances = compute_hamming_distances_batch(
                            hashes1.astype(np.uint8), hashes2.astype(np.uint8)
                        )

                        # Convert to similarities and accumulate
                        max_bits = hashes1.shape[1] * 8
                        hash_similarities = 1.0 - (distances / max_bits)

                        # Apply weight for this hash type
                        weight_map = {
                            "phash": 0.4,
                            "ahash": 0.2,
                            "dhash": 0.2,
                            "mhash": 0.1,
                        }
                        weight = weight_map.get(hash_type, 0.1)
                        similarities += hash_similarities * weight

                # Add histogram correlations if available
                if all("histogram" in fp for fp in fps1 + fps2):
                    for i in range(n1):
                        for j in range(n2):
                            hist_corr = compute_histogram_correlation(
                                fps1[i]["histogram"], fps2[j]["histogram"]
                            )
                            similarities[i, j] += max(0, hist_corr) * 0.1

                # Normalize by total weight
                total_weight = sum(weight_map.values()) + 0.1  # +0.1 for histogram
                similarities /= total_weight

                return similarities

            except Exception as e:
                logger.warning(f"Batch comparison failed, falling back: {e}")

        # Fallback to individual comparisons
        for i in range(n1):
            for j in range(n2):
                similarities[i, j] = self.compare_fingerprints(fps1[i], fps2[j])

        return similarities

    def compute_fingerprints(self, frames: np.ndarray) -> np.ndarray:
        """Compute fingerprints for multiple frames.

        Args:
            frames: Array of frames (N, H, W, C) or list of frames

        Returns:
            Array of fingerprints as feature vectors
        """
        logger.info(f"Computing fingerprints for {len(frames)} frames...")
        start_time = time.time()

        fingerprints = []

        # Process frames in batches using multiprocessing
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            # Submit all tasks
            futures = [
                executor.submit(self._compute_fingerprint_worker, frame)
                for frame in frames
            ]

            # Collect results in order
            for future in futures:
                try:
                    fingerprint = future.result()
                    # Convert fingerprint dict to feature vector
                    feature_vec = self._fingerprint_to_vector(fingerprint)
                    fingerprints.append(feature_vec)
                except Exception as e:
                    logger.warning(f"Failed to compute fingerprint: {e}")
                    # Add zero vector as fallback
                    fingerprints.append(np.zeros(self._get_fingerprint_size()))

        elapsed = time.time() - start_time
        fps = len(fingerprints) / elapsed if elapsed > 0 else 0
        logger.info(
            f"Computed {len(fingerprints)} fingerprints in {elapsed:.2f}s "
            f"({fps:.1f} fps)"
        )

        return np.array(fingerprints)

    def _fingerprint_to_vector(self, fingerprint: dict[str, np.ndarray]) -> np.ndarray:
        """Convert fingerprint dictionary to feature vector.

        Args:
            fingerprint: Dictionary of hash values and histogram

        Returns:
            Flattened feature vector
        """
        features = []

        # Extract hash values in consistent order
        for hash_name in ["phash", "ahash", "dhash", "mhash"]:
            if hash_name in fingerprint:
                features.append(fingerprint[hash_name].flatten())

        # Add histogram if present
        if "histogram" in fingerprint:
            features.append(fingerprint["histogram"])

        # Concatenate all features
        if features:
            return np.concatenate(features)
        else:
            return np.zeros(self._get_fingerprint_size())

    def _get_fingerprint_size(self) -> int:
        """Get the size of the fingerprint feature vector.

        Returns:
            Size of feature vector
        """
        # Compute size based on available hashers
        # Most hashes produce 8-byte (64-bit) values
        size = 0
        for name in ["phash", "ahash", "dhash", "mhash"]:
            if name in self.hashers:
                size += 8  # 8 bytes per hash

        # Add histogram size (32 bins * 3 channels)
        size += 32 * 3

        return size

    def clear_cache(self, video_path: str | None = None):
        """Clear fingerprint cache.

        Args:
            video_path: Specific video to clear, or None for all
        """
        if video_path:
            self.fingerprint_cache.pop(video_path, None)
        else:
            self.fingerprint_cache.clear()
