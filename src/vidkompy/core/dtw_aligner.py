#!/usr/bin/env python3
# this_file: src/vidkompy/core/dtw_aligner.py

"""
Dynamic Time Warping (DTW) for video frame alignment.

This module implements DTW algorithm for finding the globally optimal
monotonic alignment between two video sequences, preventing temporal artifacts.
"""

import numpy as np
from collections.abc import Callable
from loguru import logger
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console

from vidkompy.models import FrameAlignment
from vidkompy.core.numba_optimizations import (
    compute_dtw_cost_matrix_numba,
    find_dtw_path_numba,
    prepare_fingerprints_for_numba,
    log_numba_compilation,
)

console = Console()


class DTWAligner:
    """Dynamic Time Warping for video frame alignment.

    Why DTW over current greedy matching:
    - Guarantees monotonic alignment (no backward jumps)
    - Finds globally optimal path, not just local matches
    - Handles speed variations naturally
    - Proven algorithm from speech/time series analysis

    Why Sakoe-Chiba band constraint:
    - Reduces complexity from O(N²) to O(N×window)
    - Prevents extreme time warping
    - Makes algorithm practical for long videos
    """

    def __init__(self, window: int = 100):
        """Initialize DTW aligner with constraints.

        Args:
            window: Maximum deviation from diagonal path
                              (Sakoe-Chiba band width). Set to 0 to use default.
        """
        self.window = window
        self.default_window = 100
        self.use_numba = True  # Flag to enable/disable numba optimizations
        self._numba_compiled = False

    def set_window(self, window: int):
        """Set the window constraint for DTW alignment.

        Args:
            window: Window size for sliding frame matching. 0 means use default.
        """
        if window > 0:
            self.window = window
        else:
            self.window = self.default_window

    def align_videos(
        self,
        fg_fingerprints: dict[int, dict[str, np.ndarray]],
        bg_fingerprints: dict[int, dict[str, np.ndarray]],
        fingerprint_compare_func: Callable,
        show_progress: bool = True,
    ) -> list[tuple[int, int, float]]:
        """Find optimal monotonic alignment using DTW.

        Args:
            fg_fingerprints: Foreground video fingerprints {frame_idx: fingerprint}
            bg_fingerprints: Background video fingerprints {frame_idx: fingerprint}
            fingerprint_compare_func: Function to compare two fingerprints (0-1 similarity)
            show_progress: Whether to show progress bar

        Returns:
            List of (bg_idx, fg_idx, confidence) tuples representing optimal alignment

        Why this approach:
        - Works on pre-computed fingerprints for speed
        - Returns confidence scores for quality assessment
        - Maintains fg frame order (as required)
        """
        fg_indices = sorted(fg_fingerprints.keys())
        bg_indices = sorted(bg_fingerprints.keys())

        n_fg = len(fg_indices)
        n_bg = len(bg_indices)

        logger.info(
            f"Starting DTW alignment: {n_fg} fg frames × {n_bg} bg frames, "
            f"window={self.window}"
        )

        # Build DTW cost matrix
        dtw_matrix = self._build_dtw_matrix(
            fg_indices,
            bg_indices,
            fg_fingerprints,
            bg_fingerprints,
            fingerprint_compare_func,
            show_progress,
        )

        # Find optimal path
        path = self._find_optimal_path(dtw_matrix, n_fg, n_bg)

        # Convert path to frame alignments with confidence scores
        alignments = self._path_to_alignments(
            path,
            fg_indices,
            bg_indices,
            fg_fingerprints,
            bg_fingerprints,
            fingerprint_compare_func,
        )

        logger.info(f"DTW completed: {len(alignments)} frame alignments")

        return alignments

    def _build_dtw_matrix(
        self,
        fg_indices: list[int],
        bg_indices: list[int],
        fg_fingerprints: dict[int, dict[str, np.ndarray]],
        bg_fingerprints: dict[int, dict[str, np.ndarray]],
        compare_func: Callable,
        show_progress: bool,
    ) -> np.ndarray:
        """Build DTW cost matrix with Sakoe-Chiba band constraint.

        Why band constraint:
        - Prevents extreme time warping
        - Reduces computation from O(N²) to O(N×window)
        - Enforces reasonable temporal alignment
        """
        n_fg = len(fg_indices)
        n_bg = len(bg_indices)

        # Try to use Numba optimization if available
        if self.use_numba and n_fg > 10 and n_bg > 10:  # Only use for non-trivial sizes
            try:
                if not self._numba_compiled:
                    log_numba_compilation()
                    self._numba_compiled = True
                
                # Convert fingerprints to feature arrays for numba
                fg_features = self._fingerprints_to_features(fg_fingerprints, fg_indices)
                bg_features = self._fingerprints_to_features(bg_fingerprints, bg_indices)
                
                if show_progress:
                    console.print("  Using Numba-optimized DTW computation...")
                
                # Use numba-optimized version
                dtw = compute_dtw_cost_matrix_numba(fg_features, bg_features, self.window)
                return dtw
                
            except Exception as e:
                logger.warning(f"Failed to use Numba optimization: {e}")
                logger.info("Falling back to standard implementation")
        
        # Standard implementation
        # Initialize with infinity
        dtw = np.full((n_fg + 1, n_bg + 1), np.inf)
        dtw[0, 0] = 0

        # Progress tracking
        if show_progress:
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=True,
            )
            task = progress.add_task("  Building DTW matrix...", total=n_fg)
            progress.start()

        # Fill DTW matrix with band constraint
        for i in range(1, n_fg + 1):
            # Sakoe-Chiba band: only compute within window of diagonal
            j_start = max(1, i - self.window)
            j_end = min(n_bg + 1, i + self.window)

            for j in range(j_start, j_end):
                # Get fingerprints
                fg_fp = fg_fingerprints[fg_indices[i - 1]]
                bg_fp = bg_fingerprints[bg_indices[j - 1]]

                # Compute cost (1 - similarity)
                similarity = compare_func(fg_fp, bg_fp)
                cost = 1.0 - similarity

                # DTW recursion: min of three possible paths
                dtw[i, j] = cost + min(
                    dtw[i - 1, j],  # Skip bg frame (insertion)
                    dtw[i, j - 1],  # Skip fg frame (deletion)
                    dtw[i - 1, j - 1],  # Match frames
                )

            if show_progress:
                progress.update(task, advance=1)

        if show_progress:
            progress.stop()

        return dtw

    def _find_optimal_path(
        self, dtw: np.ndarray, n_fg: int, n_bg: int
    ) -> list[tuple[int, int]]:
        """Backtrack through DTW matrix to find optimal path.

        Why backtracking:
        - Recovers the actual alignment from cost matrix
        - Guarantees monotonic path
        - Handles insertions/deletions/matches
        """
        # Try to use Numba optimization if available
        if self.use_numba and n_fg > 10 and n_bg > 10:
            try:
                # Use numba-optimized version
                path_array = find_dtw_path_numba(dtw)
                # Convert to list of tuples
                path = [(int(i), int(j)) for i, j in path_array]
                return path
            except Exception as e:
                logger.warning(f"Failed to use Numba path finding: {e}")
                logger.info("Falling back to standard implementation")
        
        # Standard implementation
        path = []
        i, j = n_fg, n_bg

        # Backtrack from end to start
        while i > 0 or j > 0:
            path.append((i, j))

            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                # Choose direction with minimum cost
                costs = [
                    (i - 1, j, dtw[i - 1, j]),  # From above
                    (i, j - 1, dtw[i, j - 1]),  # From left
                    (i - 1, j - 1, dtw[i - 1, j - 1]),  # From diagonal
                ]

                # Filter out invalid positions
                valid_costs = [
                    (pi, pj, cost) for pi, pj, cost in costs if cost != np.inf
                ]

                if valid_costs:
                    i, j, _ = min(valid_costs, key=lambda x: x[2])
                else:
                    # Fallback: move diagonally
                    i, j = i - 1, j - 1

        # Reverse to get forward path
        path.reverse()

        # Remove dummy start position
        if path and path[0] == (0, 0):
            path = path[1:]

        return path

    def _path_to_alignments(
        self,
        path: list[tuple[int, int]],
        fg_indices: list[int],
        bg_indices: list[int],
        fg_fingerprints: dict[int, dict[str, np.ndarray]],
        bg_fingerprints: dict[int, dict[str, np.ndarray]],
        compare_func: Callable,
    ) -> list[tuple[int, int, float]]:
        """Convert DTW path to frame alignments with confidence scores.

        Why confidence scores:
        - Helps identify problematic alignments
        - Enables quality-based filtering
        - Provides feedback for debugging
        """
        alignments = []

        for i, j in path:
            # Skip boundary positions
            if i == 0 or j == 0:
                continue

            # Get actual frame indices
            fg_idx = fg_indices[i - 1]
            bg_idx = bg_indices[j - 1]

            # Compute confidence as similarity score
            fg_fp = fg_fingerprints[fg_idx]
            bg_fp = bg_fingerprints[bg_idx]
            confidence = compare_func(fg_fp, bg_fp)

            alignments.append((bg_idx, fg_idx, confidence))

        # Remove duplicate fg frames (keep best match)
        unique_alignments = {}
        for bg_idx, fg_idx, conf in alignments:
            if fg_idx not in unique_alignments or conf > unique_alignments[fg_idx][1]:
                unique_alignments[fg_idx] = (bg_idx, conf)

        # Convert back to list format
        final_alignments = [
            (bg_idx, fg_idx, conf)
            for fg_idx, (bg_idx, conf) in sorted(unique_alignments.items())
        ]

        return final_alignments

    def create_frame_alignments(
        self,
        dtw_matches: list[tuple[int, int, float]],
        total_fg_frames: int,
        total_bg_frames: int,
    ) -> list[FrameAlignment]:
        """Create complete frame-to-frame alignment from DTW matches.

        Args:
            dtw_matches: List of (bg_idx, fg_idx, confidence) from DTW
            total_fg_frames: Total number of foreground frames
            total_bg_frames: Total number of background frames

        Returns:
            List of FrameAlignment objects for every fg frame

        Why interpolation:
        - DTW may skip some frames
        - Need alignment for EVERY fg frame
        - Smooth interpolation prevents jumps
        """
        if not dtw_matches:
            # Fallback to simple linear mapping
            return self._create_linear_alignment(total_fg_frames, total_bg_frames)

        # Sort by fg index
        dtw_matches.sort(key=lambda x: x[1])

        # Create alignment for every fg frame
        alignments = []

        for fg_idx in range(total_fg_frames):
            # Find surrounding DTW matches
            prev_match = None
            next_match = None

            for match in dtw_matches:
                if match[1] <= fg_idx:
                    prev_match = match
                elif match[1] > fg_idx and next_match is None:
                    next_match = match
                    break

            # Interpolate or extrapolate
            if prev_match is None and next_match is None:
                # No matches at all - use linear
                bg_idx = int(fg_idx * total_bg_frames / total_fg_frames)
                confidence = 0.5
            elif prev_match is None:
                # Before first match - extrapolate
                bg_idx = max(0, next_match[0] - (next_match[1] - fg_idx))
                confidence = next_match[2] * 0.8
            elif next_match is None:
                # After last match - extrapolate
                bg_idx = min(
                    total_bg_frames - 1, prev_match[0] + (fg_idx - prev_match[1])
                )
                confidence = prev_match[2] * 0.8
            # Between matches - interpolate
            elif prev_match[1] == next_match[1]:
                # Same fg frame
                bg_idx = prev_match[0]
                confidence = prev_match[2]
            else:
                # Linear interpolation
                ratio = (fg_idx - prev_match[1]) / (next_match[1] - prev_match[1])
                bg_idx = int(prev_match[0] + ratio * (next_match[0] - prev_match[0]))
                confidence = prev_match[2] * (1 - ratio) + next_match[2] * ratio

            # Ensure valid bg index
            bg_idx = max(0, min(bg_idx, total_bg_frames - 1))

            alignments.append(
                FrameAlignment(
                    fg_frame_idx=fg_idx,
                    bg_frame_idx=bg_idx,
                    similarity_score=confidence,
                )
            )

        return alignments

    def _create_linear_alignment(
        self, total_fg_frames: int, total_bg_frames: int
    ) -> list[FrameAlignment]:
        """Create simple linear frame mapping as fallback.

        Why we need fallback:
        - DTW might fail on very dissimilar videos
        - Better than no alignment at all
        - Maintains temporal order
        """
        ratio = total_bg_frames / total_fg_frames if total_fg_frames > 0 else 1.0

        alignments = []
        for fg_idx in range(total_fg_frames):
            bg_idx = min(int(fg_idx * ratio), total_bg_frames - 1)

            alignments.append(
                FrameAlignment(
                    fg_frame_idx=fg_idx,
                    bg_frame_idx=bg_idx,
                    similarity_score=0.5,  # Unknown confidence
                )
            )

        return alignments

    def _compute_cost_matrix(
        self, fg_features: np.ndarray, bg_features: np.ndarray
    ) -> np.ndarray:
        """Compute cost matrix for DTW from feature arrays.

        Args:
            fg_features: Foreground feature vectors (N, D)
            bg_features: Background feature vectors (M, D)

        Returns:
            Cost matrix (N, M)
        """
        n_fg = len(fg_features)
        n_bg = len(bg_features)

        # Initialize cost matrix
        cost_matrix = np.zeros((n_fg, n_bg))

        # Compute pairwise distances
        for i in range(n_fg):
            for j in range(n_bg):
                # Euclidean distance between feature vectors
                cost_matrix[i, j] = np.linalg.norm(fg_features[i] - bg_features[j])

        # Normalize costs to [0, 1]
        if cost_matrix.max() > 0:
            cost_matrix = cost_matrix / cost_matrix.max()

        return cost_matrix

    def _compute_path(self, cost_matrix: np.ndarray) -> list[tuple[int, int]]:
        """Compute optimal DTW path from cost matrix.

        Args:
            cost_matrix: Cost matrix (N, M)

        Returns:
            List of (i, j) indices representing the optimal path
        """
        n_fg, n_bg = cost_matrix.shape

        # Initialize DTW matrix
        dtw = np.full((n_fg + 1, n_bg + 1), np.inf)
        dtw[0, 0] = 0

        # Fill DTW matrix with window constraint
        for i in range(1, n_fg + 1):
            # Sakoe-Chiba band
            j_start = max(1, i - self.window)
            j_end = min(n_bg + 1, i + self.window)

            for j in range(j_start, j_end):
                cost = cost_matrix[i - 1, j - 1]

                # DTW recursion
                dtw[i, j] = cost + min(
                    dtw[i - 1, j],  # Insertion
                    dtw[i, j - 1],  # Deletion
                    dtw[i - 1, j - 1],  # Match
                )

        # Backtrack to find path
        path = []
        i, j = n_fg, n_bg

        while i > 0 and j > 0:
            path.append((i - 1, j - 1))  # Convert to 0-based indices

            # Choose direction with minimum cost
            if i == 1:
                j -= 1
            elif j == 1:
                i -= 1
            else:
                costs = [
                    dtw[i - 1, j],  # From above
                    dtw[i, j - 1],  # From left
                    dtw[i - 1, j - 1],  # From diagonal
                ]

                min_idx = np.argmin(costs)
                if min_idx == 0:
                    i -= 1
                elif min_idx == 1:
                    j -= 1
                else:
                    i -= 1
                    j -= 1

        # Add remaining path to origin
        while i > 0:
            i -= 1
            path.append((i, 0))
        while j > 0:
            j -= 1
            path.append((0, j))

        path.reverse()

        # Remove any invalid entries
        path = [(i, j) for i, j in path if i >= 0 and j >= 0]

        return path

    def _fingerprints_to_features(
        self, fingerprints: dict[int, dict[str, np.ndarray]], indices: list[int]
    ) -> np.ndarray:
        """Convert fingerprints to feature matrix for Numba processing.
        
        Args:
            fingerprints: Dictionary of fingerprints
            indices: List of frame indices
            
        Returns:
            Feature matrix where each row is a flattened fingerprint
        """
        # Get sample fingerprint to determine feature size
        sample_fp = next(iter(fingerprints.values()))
        
        # Calculate total feature size
        feature_size = 0
        for key, value in sample_fp.items():
            if key == "histogram":
                feature_size += len(value)
            else:
                # Hash values
                feature_size += value.size
        
        # Build feature matrix
        features = np.zeros((len(indices), feature_size), dtype=np.float32)
        
        for i, idx in enumerate(indices):
            fp = fingerprints[idx]
            offset = 0
            
            # Add hash features
            for key in sorted(fp.keys()):
                if key == "histogram":
                    continue
                value = fp[key].flatten()
                features[i, offset:offset + len(value)] = value.astype(np.float32)
                offset += len(value)
            
            # Add histogram at the end
            if "histogram" in fp:
                hist = fp["histogram"]
                features[i, offset:offset + len(hist)] = hist
        
        return features
