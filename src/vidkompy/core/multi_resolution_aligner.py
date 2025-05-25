#!/usr/bin/env python3
# this_file: src/vidkompy/core/multi_resolution_aligner.py

"""
Multi-resolution temporal alignment for precise video synchronization.

Implements hierarchical DTW with progressive refinement to eliminate drift.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from loguru import logger

from .dtw_aligner import DTWAligner
from .frame_fingerprint import FrameFingerprinter


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
    drift_correction_interval: int = 100  # Reset alignment every N frames (increased from 32)
    drift_blend_factor: float = 0.85  # How much to trust original mapping vs linear interpolation


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
        for res in self.resolutions:
            indices = list(range(0, len(fingerprints), res))
            pyramid[res] = fingerprints[indices]
            logger.debug(f"Resolution 1/{res}: {len(pyramid[res])} samples")

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

        logger.info(f"Coarse alignment at 1/{coarsest_res} resolution")
        logger.debug(f"FG samples: {len(fg_coarse)}, BG samples: {len(bg_coarse)}")

        # Use DTW with large window
        window_size = int(len(bg_coarse) * self.config.initial_window_ratio)
        dtw = DTWAligner(window=window_size)

        # Compute cost matrix
        cost_matrix = dtw._compute_cost_matrix(fg_coarse, bg_coarse)

        # Find optimal path
        path = dtw._compute_path(cost_matrix)

        # Convert path to frame mapping
        mapping = np.zeros(len(fg_coarse), dtype=int)
        for i, (fg_idx, bg_idx) in enumerate(path):
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
        logger.info(f"Refining alignment: 1/{from_res} -> 1/{to_res}")

        # Get fingerprints at target resolution
        fg_fine = fg_pyramid[to_res]
        bg_fine = bg_pyramid[to_res]

        # Calculate scaling factor
        scale = from_res // to_res

        # Initialize refined mapping
        refined_mapping = np.zeros(len(fg_fine), dtype=int)

        # Refine each coarse segment
        for i in range(len(coarse_mapping)):
            # Find corresponding fine-resolution range
            fg_start = i * scale
            fg_end = min((i + 1) * scale, len(fg_fine))

            # Find search range in background
            bg_center = coarse_mapping[i] * scale
            bg_start = max(0, bg_center - self.config.refinement_window)
            bg_end = min(len(bg_fine), bg_center + self.config.refinement_window)

            if fg_end <= fg_start or bg_end <= bg_start:
                continue

            # Extract segments
            fg_segment = fg_fine[fg_start:fg_end]
            bg_segment = bg_fine[bg_start:bg_end]

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
                logger.warning(f"Local refinement failed at segment {i}: {e}")
                # Fall back to interpolation
                for j in range(fg_start, fg_end):
                    if j < len(refined_mapping):
                        refined_mapping[j] = bg_center + (j - fg_start)

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
        """Apply periodic drift correction to prevent accumulation.

        Args:
            mapping: Initial frame mapping
            interval: Correction interval (frames)

        Returns:
            Corrected frame mapping
        """
        if interval is None:
            interval = self.config.drift_correction_interval

        corrected = mapping.copy()
        num_segments = len(mapping) // interval + 1

        logger.info(f"Applying drift correction every {interval} frames")

        for seg in range(num_segments):
            start = seg * interval
            end = min((seg + 1) * interval, len(mapping))

            if start >= end:
                continue

            # Calculate expected linear progression
            expected_start = mapping[start]
            expected_end = mapping[min(end, len(mapping) - 1)]

            # Ensure monotonic progression within segment
            segment_len = end - start
            if segment_len > 1 and expected_end > expected_start:
                # Linear interpolation as baseline
                for i in range(start, end):
                    progress = (i - start) / (segment_len - 1)
                    expected = expected_start + progress * (
                        expected_end - expected_start
                    )

                    # Blend original mapping with expected progression
                    blend_factor = self.config.drift_blend_factor  # Use config value (0.85)
                    corrected[i] = int(
                        blend_factor * mapping[i] + (1 - blend_factor) * expected
                    )
                    
                    # Log significant corrections
                    drift = abs(corrected[i] - mapping[i])
                    if drift > 5 and self.verbose:
                        logger.debug(f"Frame {i}: drift correction of {drift} frames applied")

        # Ensure final mapping is monotonic
        for i in range(1, len(corrected)):
            corrected[i] = max(corrected[i], corrected[i - 1])

        return corrected

    def interpolate_full_mapping(
        self, sparse_mapping: np.ndarray, target_length: int, source_resolution: int
    ) -> np.ndarray:
        """Interpolate sparse mapping to full frame resolution.

        Args:
            sparse_mapping: Mapping at sparse resolution
            target_length: Number of frames in full video
            source_resolution: Resolution of sparse mapping

        Returns:
            Full frame-by-frame mapping
        """
        full_mapping = np.zeros(target_length, dtype=int)

        for i in range(target_length):
            # Find corresponding sparse index
            sparse_idx = i // source_resolution

            if sparse_idx >= len(sparse_mapping) - 1:
                # Use last mapping
                full_mapping[i] = sparse_mapping[-1]
            else:
                # Interpolate between sparse mappings
                alpha = (i % source_resolution) / source_resolution
                start_bg = sparse_mapping[sparse_idx]
                end_bg = sparse_mapping[sparse_idx + 1]

                full_mapping[i] = int(start_bg + alpha * (end_bg - start_bg))

        # Ensure monotonic
        for i in range(1, len(full_mapping)):
            full_mapping[i] = max(full_mapping[i], full_mapping[i - 1])

        return full_mapping

    def align(
        self,
        fg_frames: np.ndarray,
        bg_frames: np.ndarray,
        fg_fingerprints: np.ndarray | None = None,
        bg_fingerprints: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Perform multi-resolution temporal alignment.

        Args:
            fg_frames: Foreground video frames
            bg_frames: Background video frames
            fg_fingerprints: Pre-computed foreground fingerprints
            bg_fingerprints: Pre-computed background fingerprints

        Returns:
            Frame mapping and alignment confidence
        """
        logger.info(
            f"Multi-resolution alignment: {len(fg_frames)} -> {len(bg_frames)} frames"
        )

        # Create temporal pyramids
        fg_pyramid = self.create_temporal_pyramid(fg_frames, fg_fingerprints)
        bg_pyramid = self.create_temporal_pyramid(bg_frames, bg_fingerprints)

        # Perform hierarchical alignment
        sparse_mapping = self.hierarchical_alignment(fg_pyramid, bg_pyramid)

        # Apply drift correction
        corrected_mapping = self.apply_drift_correction(sparse_mapping)

        # Interpolate to full resolution
        full_mapping = self.interpolate_full_mapping(
            corrected_mapping,
            len(fg_frames),
            self.resolutions[-1],  # Finest resolution used
        )

        # Calculate confidence based on mapping smoothness
        differences = np.diff(full_mapping)
        expected_diff = len(bg_frames) / len(fg_frames)
        variance = np.var(differences)
        confidence = np.exp(-variance / (expected_diff**2))

        logger.info(f"Alignment complete. Confidence: {confidence:.3f}")

        return full_mapping, confidence
