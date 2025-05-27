#!/usr/bin/env python3
# this_file: src/vidkompy/core/multi_resolution_aligner.py

"""
Multi-resolution temporal alignment for precise video synchronization.

Implements hierarchical DTW with progressive refinement to eliminate drift.
"""

from dataclasses import dataclass
import numpy as np
from loguru import logger

# Import Savitzky-Golay filter if it's not already imported
from scipy.signal import savgol_filter

from .dtw_aligner import DTWAligner
from .frame_fingerprint import FrameFingerprinter

try:
    from vidkompy.core.numba_optimizations import (
        apply_polynomial_drift_correction,
    )

    NUMBA_AVAILABLE = True
except ImportError:
    logger.warning("Numba optimizations not available for multi-resolution alignment")
    NUMBA_AVAILABLE = False


@dataclass
class PreciseEngineConfig:
    """Configuration for precise temporal alignment engine."""

    # Sampling parameters
    max_resolutions: int = 4  # Number of resolution levels
    base_resolution: int = 16  # Coarsest sampling rate

    # DTW parameters
    initial_window_ratio: float = 0.25  # Window size for coarse DTW
    refinement_window: int = 30  # Frames for sliding window

    # Quality thresholds
    similarity_threshold: float = 0.85  # Minimum frame similarity
    confidence_threshold: float = 0.9  # High confidence threshold

    # Performance tuning
    max_frames_to_process: int = 10000  # Limit for very long videos
    # Reset alignment every N frames
    drift_correction_interval: int = 100
    # Trust in original map vs linear interpolation
    drift_blend_factor: float = 0.85

    # New parameters for enhanced drift correction & smoothing
    # Options: "linear", "polynomial", "loess" (loess deferred)
    drift_correction_model: str = "polynomial"
    poly_degree: int = 2  # Degree for polynomial regression
    # loess_frac: float = 0.25 # LOESS fraction (requires statsmodels)
    adaptive_blend_factor: bool = True  # Enable adaptive blend factor
    # Savitzky-Golay filter window (odd number)
    savitzky_golay_window: int = 21
    # Savitzky-Golay poly order (< window)
    savitzky_golay_polyorder: int = 3
    # Options: "linear", "spline" (spline deferred)
    interpolation_method: str = "linear"
    cli_dtw_window: int = 0  # DTW window size from CLI, 0 to ignore


class MultiResolutionAligner:
    """Multi-resolution temporal alignment with drift correction."""

    def __init__(
        self,
        fingerprinter: FrameFingerprinter,
        config: PreciseEngineConfig | None = None,
        verbose: bool = False,
    ):
        """Initialize multi-resolution aligner.

        Args:
            fingerprinter: Frame fingerprint generator
            config: Engine configuration
            verbose: Enable detailed logging
        """
        self.fingerprinter = fingerprinter
        self.config = config or PreciseEngineConfig()
        self.verbose = verbose
        self.use_numba = NUMBA_AVAILABLE

        # Calculate resolution levels
        self.resolutions = []
        res = self.config.base_resolution
        for _ in range(self.config.max_resolutions):
            self.resolutions.append(res)
            res = max(1, res // 2)

        logger.info(f"Multi-resolution levels: {self.resolutions}")

    def create_temporal_pyramid(
        self, frames: np.ndarray, fingerprints: np.ndarray | None = None
    ) -> dict[int, np.ndarray]:
        """Create multi-resolution temporal pyramid.

        Args:
            frames: Video frames array
            fingerprints: Pre-computed fingerprints (optional)

        Returns:
            Dictionary mapping resolution to sampled fingerprints
        """
        pyramid = {}

        # Compute fingerprints if not provided
        if fingerprints is None:
            logger.info("Computing fingerprints for temporal pyramid...")
            fingerprints = self.fingerprinter.compute_fingerprints(frames)

        # Sample at each resolution
        for res_val in self.resolutions:
            indices = list(range(0, len(fingerprints), res_val))
            pyramid[res_val] = fingerprints[indices]
            # Resolution 1/{res_val}: {len(pyramid[res_val])} samples
            logger.debug(f"Res 1/{res_val}: {len(pyramid[res_val])} samples")

        return pyramid

    def coarse_alignment(
        self, fg_pyramid: dict[int, np.ndarray], bg_pyramid: dict[int, np.ndarray]
    ) -> np.ndarray:
        """Perform coarse alignment at lowest resolution.

        Args:
            fg_pyramid: Foreground temporal pyramid
            bg_pyramid: Background temporal pyramid

        Returns:
            Initial frame mapping at coarsest resolution
        """
        # Start with coarsest resolution
        coarsest_res = max(self.resolutions)
        fg_coarse = fg_pyramid[coarsest_res]
        bg_coarse = bg_pyramid[coarsest_res]

        # Coarse alignment at 1/{coarsest_res} resolution
        logger.info(f"Coarse align at 1/{coarsest_res}")
        # FG samples: {len(fg_coarse)}, BG samples: {len(bg_coarse)}
        logger.debug(f"FG smpl: {len(fg_coarse)}, BG smpl: {len(bg_coarse)}")

        # Use DTW with large window
        window_size = int(len(bg_coarse) * self.config.initial_window_ratio)
        dtw = DTWAligner(window=window_size)

        # Compute cost matrix
        cost_matrix = dtw._compute_cost_matrix(fg_coarse, bg_coarse)

        # Find optimal path
        path = dtw._compute_path(cost_matrix)

        # Convert path to frame mapping
        mapping = np.zeros(len(fg_coarse), dtype=int)
        for _i, (fg_idx, bg_idx) in enumerate(path):
            if fg_idx < len(mapping):
                mapping[fg_idx] = bg_idx

        return mapping

    def refine_alignment(
        self,
        fg_pyramid: dict[int, np.ndarray],
        bg_pyramid: dict[int, np.ndarray],
        coarse_mapping: np.ndarray,
        from_res: int,
        to_res: int,
    ) -> np.ndarray:
        """Refine alignment from one resolution to the next.

        Args:
            fg_pyramid: Foreground temporal pyramid
            bg_pyramid: Background temporal pyramid
            coarse_mapping: Mapping at coarser resolution
            from_res: Source resolution
            to_res: Target resolution (finer)

        Returns:
            Refined frame mapping at target resolution
        """
        # Refining alignment: 1/{from_res} -> 1/{to_res}
        logger.info(f"Refine: 1/{from_res} -> 1/{to_res}")

        # Get fingerprints at target resolution
        fg_fine = fg_pyramid[to_res]
        bg_fine = bg_pyramid[to_res]

        # Calculate scaling factor
        scale = from_res // to_res

        # Initialize refined mapping
        refined_mapping = np.zeros(len(fg_fine), dtype=int)

        # Refine each coarse segment
        for i_seg in range(len(coarse_mapping)):
            # Find corresponding fine-resolution range
            fg_start = i_seg * scale
            fg_end = min((i_seg + 1) * scale, len(fg_fine))

            # Find search range in background
            bg_center = coarse_mapping[i_seg] * scale
            bg_start = max(0, bg_center - self.config.refinement_window)
            bg_search_end = min(len(bg_fine), bg_center + self.config.refinement_window)

            if fg_end <= fg_start or bg_search_end <= bg_start:
                continue

            # Extract segments
            fg_segment = fg_fine[fg_start:fg_end]
            bg_segment = bg_fine[bg_start:bg_search_end]

            # Local DTW alignment
            window = min(len(bg_segment) // 2, 10)
            dtw = DTWAligner(window=window)

            try:
                cost_matrix = dtw._compute_cost_matrix(fg_segment, bg_segment)
                path = dtw._compute_path(cost_matrix)

                # Update refined mapping
                for fg_local, bg_local in path:
                    if fg_local < len(fg_segment):
                        fg_global = fg_start + fg_local
                        bg_global = bg_start + bg_local
                        if fg_global < len(refined_mapping):
                            refined_mapping[fg_global] = bg_global
            except Exception as e:
                # Local refinement failed at segment {i_seg}: {e}
                logger.warning(f"Local refine failed at seg {i_seg}: {e}")
                # Fall back to interpolation
                for j_loop in range(fg_start, fg_end):
                    if j_loop < len(refined_mapping):
                        refined_mapping[j_loop] = bg_center + (j_loop - fg_start)

        return refined_mapping

    def hierarchical_alignment(
        self, fg_pyramid: dict[int, np.ndarray], bg_pyramid: dict[int, np.ndarray]
    ) -> np.ndarray:
        """Perform hierarchical alignment from coarse to fine.

        Args:
            fg_pyramid: Foreground temporal pyramid
            bg_pyramid: Background temporal pyramid

        Returns:
            Frame mapping at finest resolution
        """
        # Start with coarse alignment
        mapping = self.coarse_alignment(fg_pyramid, bg_pyramid)
        current_res = max(self.resolutions)

        # Progressively refine
        for next_res in sorted(self.resolutions, reverse=True)[1:]:
            mapping = self.refine_alignment(
                fg_pyramid, bg_pyramid, mapping, current_res, next_res
            )
            current_res = next_res

        return mapping

    def apply_drift_correction(
        self, mapping: np.ndarray, interval: int | None = None
    ) -> np.ndarray:
        """Apply periodic drift correction to prevent accumulation."""
        if interval is None:
            interval = self.config.drift_correction_interval

        if len(mapping) == 0:
            return mapping  # Return empty if input is empty

        logger.info(f"Drift correction every {interval} frames")

        # Try numba optimization for polynomial drift correction
        if self.use_numba and self.config.drift_correction_model == "polynomial":
            try:
                corrected = apply_polynomial_drift_correction(
                    mapping.astype(np.float64),
                    interval,
                    self.config.poly_degree,
                    self.config.drift_blend_factor,
                )
                return corrected.astype(int)
            except Exception as e:
                logger.warning(f"Numba drift correction failed: {e}")
                logger.info("Falling back to standard implementation")

        # Standard implementation
        corrected = mapping.copy()
        num_segments = len(mapping) // interval + 1

        for seg_idx in range(num_segments):
            start_idx = seg_idx * interval
            end_idx = min((seg_idx + 1) * interval, len(mapping))

            if start_idx >= end_idx:
                continue

            segment_mapping = mapping[start_idx:end_idx]
            segment_indices = np.arange(len(segment_mapping))

            expected_segment_progression: np.ndarray

            if (
                self.config.drift_correction_model == "polynomial"
                and len(segment_mapping) > self.config.poly_degree
            ):
                coeffs = np.polyfit(
                    segment_indices, segment_mapping, self.config.poly_degree
                )
                poly = np.poly1d(coeffs)
                expected_segment_progression = poly(segment_indices)
            else:  # Default to linear if polynomial fails or not selected
                expected_start_val = segment_mapping[0]
                expected_end_val = segment_mapping[-1]
                expected_segment_progression = np.linspace(
                    expected_start_val, expected_end_val, len(segment_mapping)
                )

            current_blend_factor = self.config.drift_blend_factor
            if self.config.adaptive_blend_factor and len(segment_mapping) > 1:
                # Simplified adaptive factor: less trust if variance is high
                variance = np.var(np.diff(segment_mapping))
                # Normalize variance crudely; this needs better scaling
                norm_variance = np.clip(
                    variance / (abs(np.mean(np.diff(segment_mapping))) + 1e-5), 0, 1
                )
                current_blend_factor = self.config.drift_blend_factor * (
                    1 - 0.5 * norm_variance
                )

            for k_loop in range(len(segment_mapping)):
                map_k_idx = start_idx + k_loop
                corrected[map_k_idx] = int(
                    current_blend_factor * segment_mapping[k_loop]
                    + (1 - current_blend_factor) * expected_segment_progression[k_loop]
                )
                drift_val = abs(corrected[map_k_idx] - segment_mapping[k_loop])
                if drift_val > 5 and self.verbose:
                    logger.debug(f"Frame {map_k_idx}: drift corr {drift_val} applied")

        # Ensure final mapping is monotonic
        for i_mono in range(1, len(corrected)):
            corrected[i_mono] = max(corrected[i_mono], corrected[i_mono - 1])
        return corrected

    def interpolate_full_mapping(
        self, sparse_mapping: np.ndarray, target_length: int, source_resolution: int
    ) -> np.ndarray:
        """Interpolate sparse mapping to full frame resolution."""
        full_mapping = np.zeros(target_length, dtype=int)
        if len(sparse_mapping) == 0:
            if target_length > 0:
                logger.warning("Empty sparse_mapping, returning zero mapping.")
            return full_mapping

        # Current method is linear interpolation
        if self.config.interpolation_method == "linear" or True:  # Default to linear
            for i_interp in range(target_length):
                sparse_idx = i_interp // source_resolution
                if sparse_idx >= len(sparse_mapping) - 1:
                    full_mapping[i_interp] = sparse_mapping[-1]
                else:
                    alpha = (i_interp % source_resolution) / source_resolution
                    start_bg = sparse_mapping[sparse_idx]
                    end_bg = sparse_mapping[sparse_idx + 1]
                    full_mapping[i_interp] = int(start_bg + alpha * (end_bg - start_bg))
        # else if self.config.interpolation_method == "spline": # Spline deferred
        # pass

        for i_mono in range(1, len(full_mapping)):
            full_mapping[i_mono] = max(full_mapping[i_mono], full_mapping[i_mono - 1])
        return full_mapping

    def align(
        self,
        fg_frames: np.ndarray,
        bg_frames: np.ndarray,
        fg_fingerprints: np.ndarray | None = None,
        bg_fingerprints: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Perform multi-resolution temporal alignment."""
        logger.info(f"Multi-res align: {len(fg_frames)} -> {len(bg_frames)} frames")

        fg_pyramid = self.create_temporal_pyramid(fg_frames, fg_fingerprints)
        bg_pyramid = self.create_temporal_pyramid(bg_frames, bg_fingerprints)

        sparse_mapping = self.hierarchical_alignment(fg_pyramid, bg_pyramid)

        # Apply enhanced drift correction
        corrected_mapping = self.apply_drift_correction(sparse_mapping)

        # Global smoothing using Savitzky-Golay filter
        if (
            len(corrected_mapping) > self.config.savitzky_golay_window
            and self.config.savitzky_golay_window > 0
        ):
            try:
                # Ensure polyorder is less than window_length
                polyorder = min(
                    self.config.savitzky_golay_polyorder,
                    self.config.savitzky_golay_window - 1,
                )
                polyorder = max(polyorder, 0)  # Must be non-negative

                smoothed_mapping = savgol_filter(
                    corrected_mapping,
                    window_length=self.config.savitzky_golay_window,
                    polyorder=polyorder,
                ).astype(int)
                # Ensure monotonicity after smoothing
                for i_mono in range(1, len(smoothed_mapping)):
                    smoothed_mapping[i_mono] = max(
                        smoothed_mapping[i_mono], smoothed_mapping[i_mono - 1]
                    )
                corrected_mapping = smoothed_mapping
                logger.info("Applied Savitzky-Golay smoothing to mapping.")
            except Exception as e:
                logger.warning(
                    f"Savitzky-Golay smoothing failed: {e}. Using un-smoothed corrected mapping."
                )
        else:
            logger.info(
                "Skipping Savitzky-Golay smoothing (mapping too short or window disabled)."
            )

        full_mapping = self.interpolate_full_mapping(
            corrected_mapping,
            len(fg_frames),
            self.resolutions[-1],  # Finest resolution used
        )

        if len(full_mapping) > 1:
            differences = np.diff(full_mapping)
            expected_diff = (
                len(bg_frames) / len(fg_frames) if len(fg_frames) > 0 else 1.0
            )
            variance = np.var(differences)
            confidence = np.exp(-variance / (expected_diff**2 + 1e-9))
        elif len(full_mapping) == 1 and len(fg_frames) == 1:
            confidence = 1.0
        else:
            confidence = 0.0

        logger.info(f"Alignment complete. Confidence: {confidence:.3f}")
        return full_mapping, confidence
