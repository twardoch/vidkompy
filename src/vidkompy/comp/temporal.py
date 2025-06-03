#!/usr/bin/env python3
# this_file: src/vidkompy/comp/temporal.py

"""
Temporal alignment module for synchronizing videos.

Implements frame-based temporal alignment with emphasis on
preserving all foreground frames without retiming.

"""

import numpy as np
from loguru import logger

from vidkompy.comp.data_types import VideoInfo, FrameAlignment, TemporalSync
from vidkompy.comp.video import VideoProcessor
from vidkompy.comp.tunnel import (
    TunnelFullSyncer,
    TunnelMaskSyncer,
    TunnelConfig,
)


class TemporalSyncer:
    """Handles temporal alignment between videos.

    This module synchronizes two videos in time, finding which frames
    correspond between them. This is the most complex part of vidkompy.

    Why temporal alignment is critical:
    - Videos may start at different times
    - Frame rates might differ
    - Some frames might be added/dropped in one video
    - The FG video timing must be preserved (it's the reference)

    Current implementation uses keyframe matching with interpolation.
    Future versions will use Dynamic Time Warping (see SPEC4.md).

    Used in:
    - vidkompy/comp/align.py
    """

    def __init__(
        self,
        processor: VideoProcessor,
        max_keyframes: int = 200,
        drift_interval: int = 100,
        window: int = 100,
        engine_mode: str = "fast",
    ):
        """Initialize temporal aligner.

        Args:
            processor: Video processor instance
            max_keyframes: Maximum keyframes for frame matching
            drift_interval: Frame interval for drift correction
            window: DTW window size
            engine_mode: Alignment engine ('full', 'mask')

        """
        self.processor = processor
        self.max_keyframes = max_keyframes
        self.drift_interval = drift_interval
        self.engine_mode = engine_mode
        self.use_tunnel_engine = engine_mode in {"full", "mask"}
        self.cli_window_size = window
        self.tunnel_syncer = None

    def sync_frames(
        self, bg_info: VideoInfo, fg_info: VideoInfo, trim: bool = False
    ) -> TemporalSync:
        """Align videos using frame content matching.

        This method ensures ALL foreground frames are preserved without
        retiming. It finds the optimal background frame for each foreground
        frame.

        Args:
            bg_info: Background video metadata
            fg_info: Foreground video metadata
            trim: Whether to trim to overlapping segment

        Returns:
            TemporalSync with frame mappings

        Used in:
        - vidkompy/comp/align.py
        """
        logger.info("Starting frame-based temporal alignment")

        # Use tunnel engine if enabled
        if self.use_tunnel_engine:
            logger.info(f"Using {self.engine_mode} temporal alignment engine")
            return self._sync_frames_tunnel(bg_info, fg_info, trim)

        # Fallback: should not reach here if tunnel engines are working
        logger.error("No alignment engine active - fallback to direct mapping")
        return self._create_direct_mapping(bg_info, fg_info)

    def _sync_frames_tunnel(
        self, bg_info: VideoInfo, fg_info: VideoInfo, trim: bool = False
    ) -> TemporalSync:
        """Align videos using tunnel-based direct frame comparison.

        Args:
            bg_info: Background video metadata
            fg_info: Foreground video metadata
            trim: Whether to trim to overlapping segment

        Returns:
            TemporalSync with frame mappings

        """
        # Initialize tunnel aligner if not already done
        if self.tunnel_syncer is None:
            config = TunnelConfig(
                window_size=(self.cli_window_size if self.cli_window_size > 0 else 30),
                downsample_factor=0.5,  # Downsample to 50% for faster processing
                early_stop_threshold=0.05,
                merge_strategy="confidence_weighted",
            )

            if self.engine_mode == "full":
                self.tunnel_syncer = TunnelFullSyncer(config)
            else:  # mask
                self.tunnel_syncer = TunnelMaskSyncer(config)

        # For tunnel engine, we assume spatial alignment is already done
        # The tunnel engine operates on spatially aligned frames
        # Use default values for offsets since spatial alignment is handled separately
        logger.info("Using pre-computed spatial alignment for tunnel engine...")
        x_offset = 0
        y_offset = 0

        # Extract all frames for tunnel alignment
        logger.info("Extracting all frames for tunnel alignment...")

        # For tunnel engines, we need full frames without pre-cropping
        fg_all_frames = self.processor.extract_all_frames(
            fg_info.path,
            resize_factor=0.25,  # Use 25% size for processing
        )

        bg_all_frames = self.processor.extract_all_frames(
            bg_info.path,
            resize_factor=0.25,  # Use 25% size for processing
        )

        if fg_all_frames is None or bg_all_frames is None:
            logger.error("Failed to extract frames for tunnel alignment")
            return self._create_direct_mapping(bg_info, fg_info)

        # Scale offsets for downsampled frames
        scaled_x_offset = int(x_offset * 0.25)
        scaled_y_offset = int(y_offset * 0.25)

        # Perform tunnel alignment
        logger.info(f"Performing {self.engine_mode} alignment...")
        frame_alignments, confidence = self.tunnel_syncer.sync(
            fg_all_frames,
            bg_all_frames,
            scaled_x_offset,
            scaled_y_offset,
            verbose=True,
        )

        # Calculate temporal offset from first alignment
        if frame_alignments:
            first_align = frame_alignments[0]
            offset_seconds = (first_align.bg_frame_idx / bg_info.fps) - (
                first_align.fg_frame_idx / fg_info.fps
            )
        else:
            offset_seconds = 0.0

        return TemporalSync(
            offset_seconds=offset_seconds,
            frame_alignments=frame_alignments,
            method_used=self.engine_mode,
            confidence=confidence,
        )

    def _create_direct_mapping(
        self, bg_info: VideoInfo, fg_info: VideoInfo
    ) -> TemporalSync:
        """Create simple direct frame mapping as fallback."""
        fps_ratio = bg_info.fps / fg_info.fps if fg_info.fps > 0 else 1.0

        alignments = []
        for fg_idx in range(fg_info.frame_count):
            bg_idx = int(fg_idx * fps_ratio)
            bg_idx = min(bg_idx, bg_info.frame_count - 1)

            alignments.append(
                FrameAlignment(
                    fg_frame_idx=fg_idx,
                    bg_frame_idx=bg_idx,
                    similarity_score=0.5,
                )
            )

        return TemporalSync(
            offset_seconds=0.0,
            frame_alignments=alignments,
            method_used="direct",
            confidence=0.3,
        )

    def create_border_mask(
        self,
        spatial_alignment,
        fg_info: VideoInfo,
        bg_info: VideoInfo,
        border_thickness: int = 8,
    ) -> np.ndarray:
        """Create border mask for border-based temporal alignment.

        The border mask defines the region around the foreground video edges
        where background video is visible. This is used for similarity
        comparison in border mode.

        Args:
            spatial_alignment: Result from spatial alignment containing
                             x/y offsets
            fg_info: Foreground video information
            bg_info: Background video information
            border_thickness: Thickness of border region in pixels

        Returns:
            Binary mask where 1 indicates border region, 0 indicates
            non-border

        Used in:
        - vidkompy/comp/align.py
        """
        # Get foreground position on background canvas
        x_offset = spatial_alignment.x_offset
        y_offset = spatial_alignment.y_offset
        fg_width = fg_info.width
        fg_height = fg_info.height
        bg_width = bg_info.width
        bg_height = bg_info.height

        # Create mask same size as background
        mask = np.zeros((bg_height, bg_width), dtype=np.uint8)

        # Define foreground rectangle bounds
        fg_left = x_offset
        fg_right = x_offset + fg_width
        fg_top = y_offset
        fg_bottom = y_offset + fg_height

        # Ensure bounds are within background
        fg_left = max(0, fg_left)
        fg_right = min(bg_width, fg_right)
        fg_top = max(0, fg_top)
        fg_bottom = min(bg_height, fg_bottom)

        # Define border regions based on which edges have visible background

        # Top border (if fg doesn't touch top edge)
        if fg_top > 0:
            border_top = max(0, fg_top - border_thickness)
            mask[border_top:fg_top, fg_left:fg_right] = 1

        # Bottom border (if fg doesn't touch bottom edge)
        if fg_bottom < bg_height:
            border_bottom = min(bg_height, fg_bottom + border_thickness)
            mask[fg_bottom:border_bottom, fg_left:fg_right] = 1

        # Left border (if fg doesn't touch left edge)
        if fg_left > 0:
            border_left = max(0, fg_left - border_thickness)
            mask[fg_top:fg_bottom, border_left:fg_left] = 1

        # Right border (if fg doesn't touch right edge)
        if fg_right < bg_width:
            border_right = min(bg_width, fg_right + border_thickness)
            mask[fg_top:fg_bottom, fg_right:border_right] = 1

        logger.debug(f"Created border mask: {np.sum(mask)} pixels in border region")
        return mask

    def _apply_mask_to_frame(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply binary mask to frame, setting non-masked areas to black.

        Args:
            frame: Input frame (H, W, C) or (H, W)
            mask: Binary mask (H, W) where 1 = keep, 0 = zero out

        Returns:
            Masked frame with same dimensions as input

        """
        if len(frame.shape) == 3:
            # Color frame - apply mask to all channels
            masked = frame.copy()
            for c in range(frame.shape[2]):
                masked[:, :, c] = frame[:, :, c] * mask
        else:
            # Grayscale frame
            masked = frame * mask

        return masked

    def create_blend_mask(
        self,
        spatial_alignment,
        fg_info: VideoInfo,
        bg_info: VideoInfo,
        border_thickness: int = 8,
    ) -> np.ndarray:
        """Create blend mask for smooth edge transitions.

        Creates a gradient mask that transitions from fully opaque (1.0) in the center
        of the foreground to fully transparent (0.0) at the edges where background is visible.

        Args:
            spatial_alignment: Result from spatial alignment containing x/y offsets
            fg_info: Foreground video information
            bg_info: Background video information
            border_thickness: Width of gradient transition in pixels

        Returns:
            Float mask with values 0.0-1.0 for alpha blending

        Used in:
        - vidkompy/comp/align.py
        """
        # Get foreground position on background canvas
        x_offset = spatial_alignment.x_offset
        y_offset = spatial_alignment.y_offset
        fg_width = fg_info.width
        fg_height = fg_info.height
        bg_width = bg_info.width
        bg_height = bg_info.height

        # Create mask same size as foreground (will be placed on background)
        mask = np.ones((fg_height, fg_width), dtype=np.float32)

        # Determine which edges need blending (where bg is visible)
        blend_top = y_offset > 0
        blend_bottom = (y_offset + fg_height) < bg_height
        blend_left = x_offset > 0
        blend_right = (x_offset + fg_width) < bg_width

        # Create gradient on edges that need blending
        for y in range(fg_height):
            for x in range(fg_width):
                alpha = 1.0

                # Top edge gradient
                if blend_top and y < border_thickness:
                    alpha = min(alpha, y / border_thickness)

                # Bottom edge gradient
                if blend_bottom and y >= (fg_height - border_thickness):
                    alpha = min(alpha, (fg_height - 1 - y) / border_thickness)

                # Left edge gradient
                if blend_left and x < border_thickness:
                    alpha = min(alpha, x / border_thickness)

                # Right edge gradient
                if blend_right and x >= (fg_width - border_thickness):
                    alpha = min(alpha, (fg_width - 1 - x) / border_thickness)

                mask[y, x] = max(0.0, min(1.0, alpha))

        logger.debug(f"Created blend mask with {border_thickness}px gradient")
        return mask
