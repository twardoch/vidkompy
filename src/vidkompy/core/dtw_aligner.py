#!/usr/bin/env python3
# this_file: src/vidkompy/core/dtw_aligner.py

"""
Dynamic Time Warping (DTW) for video frame alignment.

This module implements DTW algorithm for finding the globally optimal
monotonic alignment between two video sequences, preventing temporal artifacts.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from loguru import logger
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console

from ..models import FrameAlignment

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

    def __init__(self, window_constraint: int = 100):
        """Initialize DTW aligner with constraints.

        Args:
            window_constraint: Maximum deviation from diagonal path
                              (Sakoe-Chiba band width)
        """
        self.window = window_constraint

    def align_videos(
        self,
        fg_fingerprints: Dict[int, Dict[str, np.ndarray]],
        bg_fingerprints: Dict[int, Dict[str, np.ndarray]],
        fingerprint_compare_func: Callable,
        show_progress: bool = True,
    ) -> List[Tuple[int, int, float]]:
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
        fg_indices: List[int],
        bg_indices: List[int],
        fg_fingerprints: Dict[int, Dict[str, np.ndarray]],
        bg_fingerprints: Dict[int, Dict[str, np.ndarray]],
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
    ) -> List[Tuple[int, int]]:
        """Backtrack through DTW matrix to find optimal path.

        Why backtracking:
        - Recovers the actual alignment from cost matrix
        - Guarantees monotonic path
        - Handles insertions/deletions/matches
        """
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
        path: List[Tuple[int, int]],
        fg_indices: List[int],
        bg_indices: List[int],
        fg_fingerprints: Dict[int, Dict[str, np.ndarray]],
        bg_fingerprints: Dict[int, Dict[str, np.ndarray]],
        compare_func: Callable,
    ) -> List[Tuple[int, int, float]]:
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
        dtw_matches: List[Tuple[int, int, float]],
        total_fg_frames: int,
        total_bg_frames: int,
    ) -> List[FrameAlignment]:
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
            else:
                # Between matches - interpolate
                if prev_match[1] == next_match[1]:
                    # Same fg frame
                    bg_idx = prev_match[0]
                    confidence = prev_match[2]
                else:
                    # Linear interpolation
                    ratio = (fg_idx - prev_match[1]) / (next_match[1] - prev_match[1])
                    bg_idx = int(
                        prev_match[0] + ratio * (next_match[0] - prev_match[0])
                    )
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
    ) -> List[FrameAlignment]:
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
