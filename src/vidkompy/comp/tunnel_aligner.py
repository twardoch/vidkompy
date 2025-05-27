#!/usr/bin/env python3
# this_file: src/vidkompy/comp/tunnel_aligner.py
"""Tunnel-based temporal alignment using direct frame comparison with sliding windows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from vidkompy.comp.models import FrameAlignment

console = Console()


@dataclass
class TunnelConfig:
    """Configuration for tunnel-based alignment.

    Used in:
    - vidkompy/comp/temporal_alignment.py
    """

    window_size: int = 30
    downsample_factor: float = 1.0
    early_stop_threshold: float = 0.05
    merge_strategy: str = "average"  # "average" or "confidence_weighted"
    mask_threshold: int = 10  # For tunnel_mask
    mask_erosion: int = 2  # For tunnel_mask


class TunnelAligner(ABC):
    """Base class for tunnel-based temporal alignment."""

    def __init__(self, config: TunnelConfig | None = None):
        """Initialize tunnel aligner with configuration."""
        self.config = config or TunnelConfig()
        self._frame_cache: dict[tuple[str, int], np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def align(
        self,
        fg_frames: np.ndarray,
        bg_frames: np.ndarray,
        x_offset: int,
        y_offset: int,
        verbose: bool = False,
    ) -> tuple[list[FrameAlignment], float]:
        """Perform tunnel-based temporal alignment.

        Args:
            fg_frames: Foreground frames array
            bg_frames: Background frames array
            x_offset: X offset for spatial alignment
            y_offset: Y offset for spatial alignment
            verbose: Enable verbose logging

        Returns:
            Tuple of (frame alignments, confidence score)

        Used in:
        - vidkompy/comp/temporal_alignment.py
        """
        if verbose:
            logger.info(
                f"Starting tunnel alignment with {len(fg_frames)} FG and {len(bg_frames)} BG frames"
            )
            logger.info(
                f"Config: window_size={self.config.window_size}, downsample={self.config.downsample_factor}"
            )

        # Downsample frames if requested
        if self.config.downsample_factor < 1.0:
            fg_frames = self._downsample_frames(fg_frames)
            bg_frames = self._downsample_frames(bg_frames)
            if verbose:
                logger.info(
                    f"Downsampled to {len(fg_frames)} FG and {len(bg_frames)} BG frames"
                )

        # Perform forward pass
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Forward pass...", total=len(fg_frames))
            forward_mapping = self._forward_pass(
                fg_frames, bg_frames, x_offset, y_offset, progress, task, verbose
            )
            progress.update(task, completed=len(fg_frames))

        # Perform backward pass
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Backward pass...", total=len(fg_frames))
            backward_mapping = self._backward_pass(
                fg_frames, bg_frames, x_offset, y_offset, progress, task, verbose
            )
            progress.update(task, completed=len(fg_frames))

        # Merge mappings
        final_mapping, confidence = self._merge_mappings(
            forward_mapping, backward_mapping, len(fg_frames), verbose
        )

        if verbose:
            logger.info(
                f"Cache stats: {self._cache_hits} hits, {self._cache_misses} misses"
            )
            logger.info(f"Final confidence: {confidence:.3f}")

        # Convert to FrameAlignment objects
        alignments = []
        for fg_idx, bg_idx in enumerate(final_mapping):
            alignments.append(
                FrameAlignment(
                    fg_frame_idx=fg_idx,
                    bg_frame_idx=int(bg_idx),
                    similarity_score=1.0,  # Individual confidence could be computed
                )
            )

        return alignments, confidence

    def _downsample_frames(self, frames: np.ndarray) -> np.ndarray:
        """Downsample frames by the configured factor."""
        if self.config.downsample_factor >= 1.0:
            return frames

        downsampled = []
        for frame in frames:
            h, w = frame.shape[:2]
            new_h = int(h * self.config.downsample_factor)
            new_w = int(w * self.config.downsample_factor)
            downsampled.append(
                cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            )

        return np.array(downsampled)

    def _forward_pass(
        self,
        fg_frames: np.ndarray,
        bg_frames: np.ndarray,
        x_offset: int,
        y_offset: int,
        progress: Progress,
        task: int,
        verbose: bool,
    ) -> list[int]:
        """Perform forward pass from start to end."""
        mapping = []
        last_bg_idx = 0

        for fg_idx in range(len(fg_frames)):
            # Define search window
            window_start = last_bg_idx
            window_end = min(last_bg_idx + self.config.window_size, len(bg_frames))

            # Find best match in window
            best_bg_idx, best_diff = self._find_best_match(
                fg_frames[fg_idx],
                bg_frames[window_start:window_end],
                x_offset,
                y_offset,
                window_start,
            )

            mapping.append(best_bg_idx)
            last_bg_idx = best_bg_idx

            # Early stopping if perfect match found
            if best_diff < self.config.early_stop_threshold:
                last_bg_idx = max(last_bg_idx, best_bg_idx + 1)

            progress.update(task, advance=1)

            if verbose and fg_idx % 100 == 0:
                logger.debug(
                    f"Forward: FG {fg_idx} -> BG {best_bg_idx} (diff={best_diff:.3f})"
                )

        return mapping

    def _backward_pass(
        self,
        fg_frames: np.ndarray,
        bg_frames: np.ndarray,
        x_offset: int,
        y_offset: int,
        progress: Progress,
        task: int,
        verbose: bool,
    ) -> list[int]:
        """Perform backward pass from end to start."""
        mapping = [0] * len(fg_frames)
        last_bg_idx = len(bg_frames) - 1

        for fg_idx in range(len(fg_frames) - 1, -1, -1):
            # Define search window
            window_end = last_bg_idx + 1
            window_start = max(0, last_bg_idx - self.config.window_size + 1)

            # Find best match in window
            best_bg_idx, best_diff = self._find_best_match(
                fg_frames[fg_idx],
                bg_frames[window_start:window_end],
                x_offset,
                y_offset,
                window_start,
            )

            mapping[fg_idx] = best_bg_idx
            last_bg_idx = best_bg_idx

            # Early stopping if perfect match found
            if best_diff < self.config.early_stop_threshold:
                last_bg_idx = min(last_bg_idx, best_bg_idx - 1)

            progress.update(task, advance=1)

            if verbose and fg_idx % 100 == 0:
                logger.debug(
                    f"Backward: FG {fg_idx} -> BG {best_bg_idx} (diff={best_diff:.3f})"
                )

        return mapping

    def _find_best_match(
        self,
        fg_frame: np.ndarray,
        bg_window: np.ndarray,
        x_offset: int,
        y_offset: int,
        window_offset: int,
    ) -> tuple[int, float]:
        """Find best matching BG frame in window for given FG frame."""
        best_idx = 0
        best_diff = float("inf")

        for i, bg_frame in enumerate(bg_window):
            diff = self.compute_frame_difference(fg_frame, bg_frame, x_offset, y_offset)

            if diff < best_diff:
                best_diff = diff
                best_idx = i

                # Early exit if very good match found
                if diff < self.config.early_stop_threshold:
                    break

        return window_offset + best_idx, best_diff

    def _merge_mappings(
        self, forward: list[int], backward: list[int], num_frames: int, verbose: bool
    ) -> tuple[list[int], float]:
        """Merge forward and backward mappings."""
        merged = []
        total_diff = 0.0

        if self.config.merge_strategy == "average":
            # Simple average
            for i in range(num_frames):
                merged_idx = (forward[i] + backward[i]) / 2.0
                merged.append(merged_idx)
                total_diff += abs(forward[i] - backward[i])

        elif self.config.merge_strategy == "confidence_weighted":
            # Weight by consistency between forward/backward
            for i in range(num_frames):
                diff = abs(forward[i] - backward[i])
                if diff < 5:  # High consistency
                    weight = 0.5
                else:  # Lower consistency, trust forward more
                    weight = 0.7
                merged_idx = weight * forward[i] + (1 - weight) * backward[i]
                merged.append(merged_idx)
                total_diff += diff

        # Ensure monotonicity
        for i in range(1, len(merged)):
            if merged[i] <= merged[i - 1]:
                merged[i] = merged[i - 1] + 0.1

        # Compute confidence based on forward/backward consistency
        avg_diff = total_diff / num_frames if num_frames > 0 else 0
        confidence = max(0.0, 1.0 - (avg_diff / 10.0))  # Normalize to 0-1

        if verbose:
            logger.info(
                f"Merge: avg difference = {avg_diff:.2f}, confidence = {confidence:.3f}"
            )

        return merged, confidence

    @abstractmethod
    def compute_frame_difference(
        self, fg_frame: np.ndarray, bg_frame: np.ndarray, x_offset: int, y_offset: int
    ) -> float:
        """Compute difference between FG and BG frames.

        To be implemented by subclasses.

        """


class TunnelFullAligner(TunnelAligner):
    """Tunnel aligner using full frame comparison.

    Used in:
    - vidkompy/comp/temporal_alignment.py
    """

    def compute_frame_difference(
        self, fg_frame: np.ndarray, bg_frame: np.ndarray, x_offset: int, y_offset: int
    ) -> float:
        """Compute pixel-wise difference between frames."""
        fg_h, fg_w = fg_frame.shape[:2]

        # Crop BG frame to FG region
        bg_cropped = bg_frame[y_offset : y_offset + fg_h, x_offset : x_offset + fg_w]

        # Handle size mismatch
        if bg_cropped.shape[:2] != fg_frame.shape[:2]:
            return float("inf")

        # Compute pixel-wise difference
        diff = np.abs(fg_frame.astype(float) - bg_cropped.astype(float))

        # Return mean absolute difference
        return np.mean(diff)


class TunnelMaskAligner(TunnelAligner):
    """Tunnel aligner using masked frame comparison.

    Used in:
    - vidkompy/comp/temporal_alignment.py
    """

    def __init__(self, config: TunnelConfig | None = None):
        """Initialize with mask generation."""
        super().__init__(config)
        self._mask_cache: dict[tuple[int, int], np.ndarray] = {}

    def align(
        self,
        fg_frames: np.ndarray,
        bg_frames: np.ndarray,
        x_offset: int,
        y_offset: int,
        verbose: bool = False,
    ) -> tuple[list[FrameAlignment], float]:
        """Perform alignment with mask generation.

        Used in:
        - vidkompy/comp/temporal_alignment.py
        """
        # Generate mask from first few FG frames
        self._generate_mask(fg_frames[:10])

        if verbose:
            mask_coverage = np.sum(self._mask) / self._mask.size
            logger.info(f"Generated mask with {mask_coverage:.1%} coverage")

        # Perform standard alignment
        return super().align(fg_frames, bg_frames, x_offset, y_offset, verbose)

    def _generate_mask(self, sample_frames: np.ndarray):
        """Generate content mask from sample frames."""
        if len(sample_frames) == 0:
            return

        h, w = sample_frames[0].shape[:2]

        # Accumulate non-black pixels across samples
        accumulator = np.zeros((h, w), dtype=np.float32)

        for frame in sample_frames:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Add pixels above threshold
            accumulator += (gray > self.config.mask_threshold).astype(float)

        # Create mask where majority of samples had content
        mask = (accumulator > len(sample_frames) / 2).astype(np.uint8)

        # Erode to avoid edge artifacts
        if self.config.mask_erosion > 0:
            kernel = np.ones(
                (self.config.mask_erosion, self.config.mask_erosion), np.uint8
            )
            mask = cv2.erode(mask, kernel, iterations=1)

        self._mask = mask

    def compute_frame_difference(
        self, fg_frame: np.ndarray, bg_frame: np.ndarray, x_offset: int, y_offset: int
    ) -> float:
        """Compute masked pixel difference between frames."""
        fg_h, fg_w = fg_frame.shape[:2]

        # Crop BG frame to FG region
        bg_cropped = bg_frame[y_offset : y_offset + fg_h, x_offset : x_offset + fg_w]

        # Handle size mismatch
        if bg_cropped.shape[:2] != fg_frame.shape[:2]:
            return float("inf")

        # Apply mask to both frames
        if hasattr(self, "_mask") and self._mask is not None:
            # Resize mask to match current frame size if needed
            if self._mask.shape[:2] != (fg_h, fg_w):
                mask_resized = cv2.resize(
                    self._mask, (fg_w, fg_h), interpolation=cv2.INTER_NEAREST
                )
            else:
                mask_resized = self._mask

            fg_masked = fg_frame * mask_resized[..., np.newaxis]
            bg_masked = bg_cropped * mask_resized[..., np.newaxis]
            mask_sum = np.sum(mask_resized)
        else:
            # Fallback to full frame if no mask
            fg_masked = fg_frame
            bg_masked = bg_cropped
            mask_sum = fg_h * fg_w

        if mask_sum == 0:
            return float("inf")

        # Compute pixel-wise difference in masked region
        diff = np.abs(fg_masked.astype(float) - bg_masked.astype(float))

        # Return mean absolute difference normalized by mask area
        return np.sum(diff) / (
            mask_sum * fg_frame.shape[2] if len(fg_frame.shape) == 3 else mask_sum
        )
