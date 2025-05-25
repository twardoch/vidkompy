#!/usr/bin/env python3
# this_file: src/vidkompy/core/precise_temporal_alignment.py

"""
Precise temporal alignment implementation with advanced techniques.

Combines multi-resolution alignment, keyframe anchoring, and bidirectional DTW.
"""

from typing import Tuple, Optional, List, Dict
import numpy as np
from loguru import logger
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from .frame_fingerprint import FrameFingerprinter
from .multi_resolution_aligner import MultiResolutionAligner, PreciseEngineConfig
from .dtw_aligner import DTWAligner


class PreciseTemporalAlignment:
    """Precise temporal alignment with drift elimination."""

    def __init__(self, fingerprinter: FrameFingerprinter, verbose: bool = False):
        """Initialize precise temporal alignment.

        Args:
            fingerprinter: Frame fingerprint generator
            verbose: Enable detailed logging
        """
        self.fingerprinter = fingerprinter
        self.verbose = verbose

        # Initialize multi-resolution aligner
        config = PreciseEngineConfig(
            max_resolutions=4, base_resolution=16, drift_correction_interval=100
        )
        self.multi_res_aligner = MultiResolutionAligner(fingerprinter, config, verbose)

    def detect_keyframes(
        self, fingerprints: np.ndarray, min_distance: int = 30
    ) -> np.ndarray:
        """Detect keyframes based on temporal changes.

        Args:
            fingerprints: Frame fingerprints
            min_distance: Minimum distance between keyframes

        Returns:
            Indices of detected keyframes
        """
        # Calculate temporal differences
        diffs = np.zeros(len(fingerprints) - 1)
        for i in range(len(fingerprints) - 1):
            diffs[i] = np.linalg.norm(fingerprints[i + 1] - fingerprints[i])

        # Smooth differences
        smoothed = gaussian_filter1d(diffs, sigma=3)

        # Find peaks (scene changes, motion peaks)
        peaks, properties = find_peaks(
            smoothed, distance=min_distance, prominence=np.std(smoothed) * 0.5
        )

        # Always include first and last frames
        keyframes = [0] + list(peaks) + [len(fingerprints) - 1]
        keyframes = sorted(list(set(keyframes)))

        logger.info(f"Detected {len(keyframes)} keyframes")
        return np.array(keyframes)

    def align_keyframes(
        self,
        fg_keyframes: np.ndarray,
        bg_keyframes: np.ndarray,
        fg_fingerprints: np.ndarray,
        bg_fingerprints: np.ndarray,
    ) -> dict[int, int]:
        """Align keyframes between videos.

        Args:
            fg_keyframes: Foreground keyframe indices
            bg_keyframes: Background keyframe indices
            fg_fingerprints: Foreground fingerprints
            bg_fingerprints: Background fingerprints

        Returns:
            Mapping of foreground to background keyframe indices
        """
        # Extract keyframe fingerprints
        fg_kf_prints = fg_fingerprints[fg_keyframes]
        bg_kf_prints = bg_fingerprints[bg_keyframes]

        # Use DTW for keyframe alignment
        dtw = DTWAligner(window_constraint=len(bg_keyframes))
        cost_matrix = dtw._compute_cost_matrix(fg_kf_prints, bg_kf_prints)
        path = dtw._compute_path(cost_matrix)

        # Convert to keyframe mapping
        kf_mapping = {}
        for fg_idx, bg_idx in path:
            if fg_idx < len(fg_keyframes) and bg_idx < len(bg_keyframes):
                kf_mapping[fg_keyframes[fg_idx]] = bg_keyframes[bg_idx]

        logger.info(f"Aligned {len(kf_mapping)} keyframe pairs")
        return kf_mapping

    def bidirectional_dtw(
        self,
        fg_fingerprints: np.ndarray,
        bg_fingerprints: np.ndarray,
        window: int = 100,
    ) -> np.ndarray:
        """Perform bidirectional DTW alignment.

        Args:
            fg_fingerprints: Foreground fingerprints
            bg_fingerprints: Background fingerprints
            window: DTW window size

        Returns:
            Averaged bidirectional alignment
        """
        dtw = DTWAligner(window_constraint=window)

        # Forward alignment
        logger.debug("Computing forward DTW alignment")
        cost_forward = dtw._compute_cost_matrix(fg_fingerprints, bg_fingerprints)
        path_forward = dtw._compute_path(cost_forward)

        # Backward alignment
        logger.debug("Computing backward DTW alignment")
        cost_backward = dtw._compute_cost_matrix(
            bg_fingerprints[::-1], fg_fingerprints[::-1]
        )
        path_backward = dtw._compute_path(cost_backward)

        # Convert paths to mappings
        forward_mapping = np.zeros(len(fg_fingerprints), dtype=int)
        for fg_idx, bg_idx in path_forward:
            if fg_idx < len(forward_mapping):
                forward_mapping[fg_idx] = bg_idx

        backward_mapping = np.zeros(len(fg_fingerprints), dtype=int)
        for bg_idx, fg_idx in path_backward:
            # Reverse indices
            bg_idx = len(bg_fingerprints) - 1 - bg_idx
            fg_idx = len(fg_fingerprints) - 1 - fg_idx
            if fg_idx >= 0 and fg_idx < len(backward_mapping):
                backward_mapping[fg_idx] = bg_idx

        # Average the mappings
        averaged_mapping = (forward_mapping + backward_mapping) // 2

        # Ensure monotonicity
        for i in range(1, len(averaged_mapping)):
            averaged_mapping[i] = max(averaged_mapping[i], averaged_mapping[i - 1])

        return averaged_mapping

    def refine_with_sliding_window(
        self,
        initial_mapping: np.ndarray,
        fg_fingerprints: np.ndarray,
        bg_fingerprints: np.ndarray,
        window_size: int = 30,
        search_range: int = 10,
    ) -> np.ndarray:
        """Refine alignment using sliding window approach.

        Args:
            initial_mapping: Initial frame mapping
            fg_fingerprints: Foreground fingerprints
            bg_fingerprints: Background fingerprints
            window_size: Size of sliding window
            search_range: Search range for refinement

        Returns:
            Refined frame mapping
        """
        refined_mapping = initial_mapping.copy()
        num_windows = len(fg_fingerprints) // window_size + 1

        logger.info(f"Sliding window refinement: {num_windows} windows")

        for w in range(num_windows):
            start = w * window_size
            end = min((w + 1) * window_size, len(fg_fingerprints))

            if end <= start:
                continue

            # Get window fingerprints
            fg_window = fg_fingerprints[start:end]

            # Determine search range in background
            bg_center = initial_mapping[start]
            bg_start = max(0, bg_center - search_range)
            bg_end = min(len(bg_fingerprints), bg_center + search_range + window_size)

            if bg_end <= bg_start:
                continue

            bg_window = bg_fingerprints[bg_start:bg_end]

            # Find best alignment within window
            best_offset = 0
            best_score = float("inf")

            for offset in range(
                min(search_range * 2, len(bg_window) - len(fg_window) + 1)
            ):
                score = 0
                for i in range(len(fg_window)):
                    if offset + i < len(bg_window):
                        score += np.linalg.norm(fg_window[i] - bg_window[offset + i])

                if score < best_score:
                    best_score = score
                    best_offset = offset

            # Apply refinement
            for i in range(start, end):
                if i < len(refined_mapping):
                    refined_mapping[i] = bg_start + best_offset + (i - start)

        # Smooth transitions between windows
        refined_mapping = gaussian_filter1d(refined_mapping.astype(float), sigma=5)
        refined_mapping = refined_mapping.astype(int)

        # Ensure monotonicity
        for i in range(1, len(refined_mapping)):
            refined_mapping[i] = max(refined_mapping[i], refined_mapping[i - 1])

        return refined_mapping

    def compute_alignment_confidence(
        self,
        mapping: np.ndarray,
        fg_fingerprints: np.ndarray,
        bg_fingerprints: np.ndarray,
    ) -> float:
        """Compute confidence score for alignment.

        Args:
            mapping: Frame mapping
            fg_fingerprints: Foreground fingerprints
            bg_fingerprints: Background fingerprints

        Returns:
            Confidence score (0-1)
        """
        similarities = []

        for fg_idx, bg_idx in enumerate(mapping):
            if bg_idx < len(bg_fingerprints):
                sim = 1.0 - np.linalg.norm(
                    fg_fingerprints[fg_idx] - bg_fingerprints[bg_idx]
                ) / (np.linalg.norm(fg_fingerprints[fg_idx]) + 1e-8)
                similarities.append(sim)

        if similarities:
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            # High confidence = high mean similarity and low variance
            confidence = mean_sim * np.exp(-std_sim)
            return float(np.clip(confidence, 0, 1))

        return 0.0

    def align(
        self, fg_frames: np.ndarray, bg_frames: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Perform precise temporal alignment.

        Args:
            fg_frames: Foreground video frames
            bg_frames: Background video frames

        Returns:
            Frame mapping and alignment confidence
        """
        logger.info("Starting precise temporal alignment")
        logger.info(f"FG: {len(fg_frames)} frames, BG: {len(bg_frames)} frames")

        # Compute fingerprints
        logger.info("Computing frame fingerprints...")
        fg_fingerprints = self.fingerprinter.compute_fingerprints(fg_frames)
        bg_fingerprints = self.fingerprinter.compute_fingerprints(bg_frames)

        # Method 1: Multi-resolution alignment
        logger.info("Phase 1: Multi-resolution alignment")
        multi_res_mapping, multi_res_conf = self.multi_res_aligner.align(
            fg_frames, bg_frames, fg_fingerprints, bg_fingerprints
        )

        # Method 2: Keyframe-based alignment
        logger.info("Phase 2: Keyframe detection and alignment")
        fg_keyframes = self.detect_keyframes(fg_fingerprints)
        bg_keyframes = self.detect_keyframes(bg_fingerprints)
        keyframe_mapping = self.align_keyframes(
            fg_keyframes, bg_keyframes, fg_fingerprints, bg_fingerprints
        )

        # Method 3: Bidirectional DTW on reduced samples
        logger.info("Phase 3: Bidirectional DTW refinement")
        sample_rate = max(1, len(fg_frames) // 500)  # Max 500 samples
        fg_sampled = fg_fingerprints[::sample_rate]
        bg_sampled = bg_fingerprints[::sample_rate]

        bidirectional_mapping = self.bidirectional_dtw(
            fg_sampled, bg_sampled, window=50
        )

        # Interpolate bidirectional mapping to full resolution
        full_bidir_mapping = np.interp(
            np.arange(len(fg_frames)),
            np.arange(len(bidirectional_mapping)) * sample_rate,
            bidirectional_mapping * sample_rate,
        ).astype(int)

        # Combine methods with weighted average
        combined_mapping = (0.5 * multi_res_mapping + 0.5 * full_bidir_mapping).astype(
            int
        )

        # Apply keyframe constraints
        for fg_kf, bg_kf in keyframe_mapping.items():
            combined_mapping[fg_kf] = bg_kf

        # Interpolate between keyframes
        keyframe_indices = sorted(keyframe_mapping.keys())
        for i in range(len(keyframe_indices) - 1):
            start_fg = keyframe_indices[i]
            end_fg = keyframe_indices[i + 1]
            start_bg = keyframe_mapping[start_fg]
            end_bg = keyframe_mapping[end_fg]

            # Linear interpolation between keyframes
            for j in range(start_fg + 1, end_fg):
                alpha = (j - start_fg) / (end_fg - start_fg)
                combined_mapping[j] = int(start_bg + alpha * (end_bg - start_bg))

        # Phase 4: Sliding window refinement
        logger.info("Phase 4: Sliding window refinement")
        refined_mapping = self.refine_with_sliding_window(
            combined_mapping, fg_fingerprints, bg_fingerprints
        )

        # Ensure final mapping is within bounds
        refined_mapping = np.clip(refined_mapping, 0, len(bg_frames) - 1)

        # Compute final confidence
        confidence = self.compute_alignment_confidence(
            refined_mapping, fg_fingerprints, bg_fingerprints
        )

        logger.info(f"Precise alignment complete. Confidence: {confidence:.3f}")

        return refined_mapping, confidence
