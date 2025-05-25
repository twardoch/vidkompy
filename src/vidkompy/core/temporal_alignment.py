#!/usr/bin/env python3
# this_file: src/vidkompy/core/temporal_alignment.py

"""
Temporal alignment module for synchronizing videos.

Implements frame-based temporal alignment with emphasis on
preserving all foreground frames without retiming.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from loguru import logger
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)
from rich.console import Console
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from vidkompy.models import VideoInfo, FrameAlignment, TemporalAlignment
from .video_processor import VideoProcessor
from .frame_fingerprint import FrameFingerprinter
from .dtw_aligner import DTWAligner
from .precise_temporal_alignment import PreciseTemporalAlignment
from .spatial_alignment import SpatialAligner

console = Console()


class TemporalAligner:
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
            engine_mode: Alignment engine ('fast', 'precise', 'mask')
        """
        self.processor = processor
        self.max_keyframes = max_keyframes
        self.drift_interval = drift_interval
        self.use_perceptual_hash = True
        # Cache for pHash np.ndarray values for classic keyframe matching
        self.hash_cache: dict[str, dict[int, np.ndarray]] = {}
        self._current_mask: np.ndarray | None = None
        self.engine_mode = engine_mode
        self.use_precise_engine = engine_mode == "precise" or engine_mode == "mask"
        self.cli_window_size = window

        self.fingerprinter: FrameFingerprinter | None = None
        self.dtw_aligner = DTWAligner(window=window)
        self.use_dtw = True
        self.hasher: cv2.img_hash.PHash | None = None  # type: ignore
        self.precise_aligner = None

        try:
            if hasattr(cv2, "img_hash") and hasattr(cv2.img_hash, "PHash_create"):
                self.hasher = cv2.img_hash.PHash_create()  # type: ignore
                if self.hasher is not None:
                    logger.info("✓ Perceptual hashing enabled (pHash)")
                    self.use_perceptual_hash = True
                else:
                    # cv2.img_hash.PHash_create() returned None.
                    # Perceptual hashing disabled. Falling back to SSIM.
                    logger.warning("pHash_create() is None. Fallback to SSIM.")
                    self.use_perceptual_hash = False
            else:
                # cv2.img_hash.PHash_create not found.
                # Perceptual hashing disabled. Falling back to SSIM.
                logger.warning("pHash_create not found. Fallback to SSIM.")
                self.use_perceptual_hash = False
        except AttributeError:
            # cv2.img_hash module or PHash_create not available.
            # Ensure opencv-contrib-python is correctly installed.
            # Perceptual hashing disabled. Falling back to SSIM.
            logger.warning("cv2.img_hash missing. Fallback to SSIM.")
            self.use_perceptual_hash = False
        except Exception as e:
            logger.error(f"Error initializing perceptual hasher: {e}")
            self.use_perceptual_hash = False

        if self.hasher is None and self.use_perceptual_hash:
            logger.warning("Hasher is None, forcing SSIM.")
            self.use_perceptual_hash = False

    def calculate_adaptive_keyframe_count(
        self, fg_info: VideoInfo, bg_info: VideoInfo, target_drift_frames: float = 1.0
    ) -> int:
        """Calculate optimal keyframe count to prevent drift.

        Args:
            fg_info: Foreground video info
            bg_info: Background video info
            target_drift_frames: Maximum acceptable drift in frames

        Returns:
            Optimal number of keyframes
        """
        # Account for FPS difference
        fps_ratio = abs(bg_info.fps - fg_info.fps) / max(bg_info.fps, fg_info.fps)

        # More keyframes needed for higher FPS mismatch
        fps_factor = 1.0 + fps_ratio * 2.0

        # Calculate base requirement to keep interpolation gaps small
        base_keyframes = fg_info.frame_count / (target_drift_frames * 10)

        # Apply factors
        required_keyframes = int(base_keyframes * fps_factor)

        # Clamp to reasonable range
        return max(50, min(required_keyframes, fg_info.frame_count // 2))

    def align_frames(
        self, bg_info: VideoInfo, fg_info: VideoInfo, trim: bool = False
    ) -> TemporalAlignment:
        """Align videos using frame content matching.

        This method ensures ALL foreground frames are preserved without
        retiming. It finds the optimal background frame for each foreground
        frame.

        Args:
            bg_info: Background video metadata
            fg_info: Foreground video metadata
            trim: Whether to trim to overlapping segment

        Returns:
            TemporalAlignment with frame mappings
        """
        logger.info("Starting frame-based temporal alignment")

        # Use precise engine if enabled
        if self.use_precise_engine:
            logger.info("Using precise temporal alignment engine")
            return self._align_frames_precise(bg_info, fg_info, trim)

        # Use DTW if enabled (default)
        if self.use_dtw:
            return self._align_frames_dtw(bg_info, fg_info, trim)

        # Otherwise use original keyframe matching
        keyframe_matches = self._find_keyframe_matches(bg_info, fg_info)

        if not keyframe_matches:
            logger.warning("No keyframe matches found, using direct mapping")
            # Fallback to simple frame mapping
            return self._create_direct_mapping(bg_info, fg_info)

        # Build complete frame alignment preserving all FG frames
        frame_alignments = self._build_frame_alignments(
            bg_info, fg_info, keyframe_matches, trim
        )

        # Calculate overall temporal offset
        first_match = keyframe_matches[0]
        offset_seconds = (first_match[0] / bg_info.fps) - (first_match[1] / fg_info.fps)

        return TemporalAlignment(
            offset_seconds=offset_seconds,
            frame_alignments=frame_alignments,
            method_used="frames",
            confidence=self._calculate_alignment_confidence(keyframe_matches),
        )

    def _align_frames_precise(
        self, bg_info: VideoInfo, fg_info: VideoInfo, trim: bool = False
    ) -> TemporalAlignment:
        """Align videos using the precise multi-resolution engine.

        This method uses advanced techniques including:
        - Multi-resolution temporal pyramids
        - Keyframe anchoring
        - Bidirectional DTW
        - Sliding window refinement

        Args:
            bg_info: Background video metadata
            fg_info: Foreground video metadata
            trim: Whether to trim to overlapping segment

        Returns:
            TemporalAlignment with frame mappings
        """
        # Initialize precise aligner if not already done
        if self.precise_aligner is None:
            if self.fingerprinter is None:
                self.fingerprinter = FrameFingerprinter()
            self.precise_aligner = PreciseTemporalAlignment(
                fingerprinter=self.fingerprinter,
                verbose=False,  # Use default verbosity
                interval=self.drift_interval,
                cli_window_size=self.cli_window_size,
            )

        # Perform spatial alignment first to get crop coordinates
        logger.info("Performing spatial alignment to determine crop region...")
        spatial_aligner = SpatialAligner()

        # Extract sample frames for spatial alignment
        bg_frames = self.processor.extract_frames(
            bg_info.path, [bg_info.frame_count // 2]
        )
        fg_frames = self.processor.extract_frames(
            fg_info.path, [fg_info.frame_count // 2]
        )

        if not bg_frames or not fg_frames:
            logger.error("Failed to extract sample frames for spatial alignment")
            return self._create_direct_mapping(bg_info, fg_info)

        bg_sample = bg_frames[0]
        fg_sample = fg_frames[0]

        if bg_sample is None or fg_sample is None:
            logger.error("Failed to extract sample frames for spatial alignment")
            return self._create_direct_mapping(bg_info, fg_info)

        # Get spatial alignment
        spatial_result = spatial_aligner.align(bg_sample, fg_sample)
        x_offset = spatial_result.x_offset
        y_offset = spatial_result.y_offset

        logger.info(
            f"Spatial alignment: offset=({x_offset}, {y_offset}), confidence={spatial_result.confidence:.3f}"
        )

        # Extract all frames for precise alignment
        logger.info("Extracting frames for precise alignment...")

        # For background, extract with cropping to match foreground region
        crop_region = (x_offset, y_offset, fg_info.width, fg_info.height)
        bg_frames = self.processor.extract_all_frames(
            bg_info.path, resize_factor=0.25, crop=crop_region
        )

        # For foreground, extract without cropping
        fg_frames = self.processor.extract_all_frames(fg_info.path, resize_factor=0.25)

        if bg_frames is None or fg_frames is None:
            logger.error("Failed to extract frames")
            return self._create_direct_mapping(bg_info, fg_info)

        # Perform precise alignment
        try:
            frame_mapping, confidence = self.precise_aligner.align(fg_frames, bg_frames)
        except Exception as e:
            logger.error(f"Precise alignment failed: {e}")
            logger.info("Falling back to standard alignment")
            return self._align_frames_dtw(bg_info, fg_info, trim)

        # Convert mapping to frame alignments
        frame_alignments = []

        # Determine range based on trim flag
        if trim and len(frame_mapping) > 0:
            # Find valid range where background frames are available
            start_idx = 0
            end_idx = len(frame_mapping)

            # Find first valid mapping
            for i in range(len(frame_mapping)):
                if frame_mapping[i] >= 0:
                    start_idx = i
                    break

            # Find last valid mapping
            for i in range(len(frame_mapping) - 1, -1, -1):
                if frame_mapping[i] < bg_info.frame_count:
                    end_idx = i + 1
                    break
        else:
            start_idx = 0
            end_idx = len(frame_mapping)

        # Build frame alignments
        for fg_idx in range(start_idx, end_idx):
            if fg_idx < len(frame_mapping):
                bg_idx = int(frame_mapping[fg_idx])
                # Ensure bg_idx is within bounds
                bg_idx = max(0, min(bg_idx, bg_info.frame_count - 1))

                frame_alignments.append(
                    FrameAlignment(
                        fg_frame_idx=fg_idx,
                        bg_frame_idx=bg_idx,
                        similarity_score=confidence,  # Use overall confidence
                    )
                )

        # Calculate temporal offset
        if frame_alignments:
            first_align = frame_alignments[0]
            offset_seconds = (first_align.bg_frame_idx / bg_info.fps) - (
                first_align.fg_frame_idx / fg_info.fps
            )
        else:
            offset_seconds = 0.0

        logger.info(
            f"Precise alignment complete. Mapped {len(frame_alignments)} frames with confidence {confidence:.3f}"
        )

        return TemporalAlignment(
            offset_seconds=offset_seconds,
            frame_alignments=frame_alignments,
            method_used="precise",
            confidence=confidence,
        )

    def _precompute_frame_hashes(
        self, video_path: str, frame_indices: list[int], resize_factor: float = 0.125
    ) -> dict[int, np.ndarray]:
        """Pre-compute perceptual hashes for frames in parallel."""
        if not self.use_perceptual_hash or self.hasher is None:
            return {}

        # Check cache first
        if video_path in self.hash_cache and all(
            idx in self.hash_cache[video_path] for idx in frame_indices
        ):
            return {
                idx: self.hash_cache[video_path][idx]
                for idx in frame_indices
                if idx in self.hash_cache[video_path]
            }

        logger.debug(f"Pre-computing hashes for {len(frame_indices)} frames")
        start_time = time.time()

        # Extract frames
        frames = self.processor.extract_frames(video_path, frame_indices, resize_factor)

        # Compute hashes in parallel
        hashes: dict[int, np.ndarray] = {}
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            future_to_idx = {
                executor.submit(self._compute_frame_hash, frame): idx
                for idx, frame in zip(frame_indices, frames, strict=False)
                if frame is not None
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    hash_value = future.result()  # This should be np.ndarray (pHash)
                    if hash_value is not None:
                        hashes[idx] = hash_value
                except Exception as e:
                    logger.warning(f"Failed to compute hash for frame {idx}: {e}")

        # Update cache (ensure it stores np.ndarray for pHash)
        if video_path not in self.hash_cache:
            self.hash_cache[video_path] = {}
        # This logic seems to intend to store full fingerprint dicts, but for cv2.norm we need pHash np.array
        # For _find_keyframe_matches, we need the pHash directly. Let's ensure this cache stores that.
        self.hash_cache[video_path].update(
            hashes
        )  # hashes should be dict[int, np.ndarray]

        elapsed = time.time() - start_time
        logger.debug(f"Computed {len(hashes)} hashes in {elapsed:.2f}s")

        return hashes

    def _precompute_masked_video_fingerprints(
        self,
        video_path: str,
        frame_indices: list[int],
        mask: np.ndarray,
        resize_factor: float = 0.25,
    ) -> dict[int, dict[str, np.ndarray]]:
        """Pre-compute masked fingerprints for video frames.

        Args:
            video_path: Path to video file
            frame_indices: List of frame indices to process
            mask: Binary mask for border regions
            resize_factor: Factor to resize frames before fingerprinting

        Returns:
            Dictionary mapping frame indices to fingerprints
        """
        if self.fingerprinter is None:
            msg = "Fingerprinter not initialized"
            raise RuntimeError(msg)

        logger.debug(f"Computing masked fingerprints for {len(frame_indices)} frames")
        start_time = time.time()

        # Extract frames
        frames = self.processor.extract_frames(video_path, frame_indices, resize_factor)

        # Compute masked fingerprints
        fingerprints = {}
        for idx, frame in zip(frame_indices, frames, strict=False):
            if frame is not None:
                # Resize mask to match frame size if needed
                if frame.shape[:2] != mask.shape[:2]:
                    resized_mask = cv2.resize(
                        mask.astype(np.float32),
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    resized_mask = (resized_mask > 0.5).astype(np.uint8)
                else:
                    resized_mask = mask

                # Compute masked fingerprint
                fingerprint = self.fingerprinter.compute_masked_fingerprint(
                    frame, resized_mask
                )
                fingerprints[idx] = fingerprint

        elapsed = time.time() - start_time
        logger.debug(
            f"Computed {len(fingerprints)} masked fingerprints in {elapsed:.2f}s"
        )

        return fingerprints

    def _find_keyframe_matches(
        self, bg_info: VideoInfo, fg_info: VideoInfo
    ) -> list[tuple[int, int, float]]:
        """Find matching keyframes between videos using monotonic dynamic programming.

        Returns:
            List of (bg_frame_idx, fg_frame_idx, similarity) tuples

        Why keyframe matching:
        - Can't compare all frame pairs (too expensive)
        - Keyframes capture important moments
        - Interpolation fills in between keyframes

        Why uniform sampling:
        - Ensures coverage of entire video
        - Predictable behavior
        - Works for any content type

        Current issues (see SPEC4.md for solutions):
        - Independent matching breaks monotonicity
        - Cost matrix computation is slow
        - Poor interpolation between sparse keyframes
        """
        # Use adaptive keyframe density calculation
        adaptive_keyframes = self.calculate_adaptive_keyframe_count(fg_info, bg_info)
        effective_target_keyframes = min(
            self.max_keyframes, adaptive_keyframes, fg_info.frame_count
        )

        logger.info(f"Adaptive calculation suggests {adaptive_keyframes} keyframes")
        logger.info(
            f"Using {effective_target_keyframes} keyframes (clamped by max_keyframes={self.max_keyframes})"
        )

        if (
            not self.use_perceptual_hash or self.hasher is None
        ):  # Indicates SSIM will be used
            # If SSIM is used and the number of keyframes is high, warn the user.
            if effective_target_keyframes > 200:  # Threshold for SSIM warning
                logger.warning(
                    f"SSIM mode with high target of {effective_target_keyframes} keyframes."
                )
            else:
                logger.info(
                    f"SSIM mode. Target keyframes: {effective_target_keyframes}"
                )
        else:
            logger.info(
                f"Perceptual hash mode. Target keyframes: {effective_target_keyframes}"
            )

        sample_interval = max(1, fg_info.frame_count // effective_target_keyframes)

        logger.info(f"Sampling every {sample_interval} frames for keyframe matching")

        # Prepare indices - sample uniformly across the video
        fg_indices = list(range(0, fg_info.frame_count, sample_interval))
        # Always include first and last frames
        if 0 not in fg_indices:
            fg_indices.insert(0, 0)
        if fg_info.frame_count - 1 not in fg_indices:
            fg_indices.append(fg_info.frame_count - 1)

        # Sample more densely from background to allow flexibility
        bg_sample_interval = max(1, sample_interval // 2)
        bg_indices = list(range(0, bg_info.frame_count, bg_sample_interval))

        logger.info(
            f"Sampling {len(fg_indices)} FG frames and {len(bg_indices)} BG frames"
        )

        # Pre-compute all hashes if available
        if self.use_perceptual_hash and self.hasher is not None:
            logger.info("Pre-computing perceptual hashes...")

            # Check if we need masked hashing
            if self._current_mask is not None:
                logger.info("Computing masked perceptual hashes for border mode...")
                fg_hashes = self._precompute_masked_frame_hashes(
                    fg_info.path, fg_indices, 0.125
                )
                bg_hashes = self._precompute_masked_frame_hashes(
                    bg_info.path, bg_indices, 0.125
                )
            else:
                fg_hashes = self._precompute_frame_hashes(
                    fg_info.path, fg_indices, 0.125
                )
                bg_hashes = self._precompute_frame_hashes(
                    bg_info.path, bg_indices, 0.125
                )

            if not fg_hashes or not bg_hashes:
                logger.error("Failed to compute hashes, falling back to SSIM")
                self.use_perceptual_hash = False

        # Build cost matrix using dynamic programming approach
        logger.info("Building cost matrix for dynamic programming alignment...")
        cost_matrix = self._build_cost_matrix(bg_info, fg_info, bg_indices, fg_indices)

        if cost_matrix is None:
            logger.error("Failed to build cost matrix")
            return []

        # Find optimal monotonic path through cost matrix
        matches = self._find_optimal_path(cost_matrix, bg_indices, fg_indices)

        # Validate matches with higher quality check if needed
        if len(matches) < 10:
            logger.warning(
                f"Only found {len(matches)} matches, attempting refinement..."
            )
            matches = self._refine_matches(
                matches, bg_info, fg_info, bg_indices, fg_indices
            )

        logger.info(f"Found {len(matches)} monotonic keyframe matches")

        if self.use_perceptual_hash:
            logger.info("✓ Perceptual hashing provided significant speedup")

        return matches

    def _compute_frame_similarity(
        self, frame1: np.ndarray, frame2: np.ndarray, mask: np.ndarray | None = None
    ) -> float:
        """Compute similarity between two frames using perceptual hash or SSIM.

        Args:
            frame1: First frame to compare
            frame2: Second frame to compare
            mask: Optional binary mask to restrict comparison to specific regions
        """
        if mask is not None:
            frame1 = self._apply_mask_to_frame(frame1, mask)
            frame2 = self._apply_mask_to_frame(frame2, mask)

        if self.use_perceptual_hash and self.hasher is not None:
            # _compute_frame_hash now directly returns the pHash np.ndarray
            hash1_val = self._compute_frame_hash(frame1)
            hash2_val = self._compute_frame_hash(frame2)
            if hash1_val is None or hash2_val is None:
                return 0.0  # Or handle error appropriately
            distance = cv2.norm(hash1_val, hash2_val, cv2.NORM_HAMMING)
            max_distance = 64  # pHash is typically 64-bit
            similarity = 1.0 - (distance / max_distance)
            return float(similarity)
        else:
            return self._compute_ssim_similarity(frame1, frame2)

    def _compute_ssim_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute similarity between two frames using SSIM."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Ensure same size
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

        # Compute SSIM
        score, _ = ssim(gray1, gray2, full=True)
        return float(score)

    def _precompute_masked_frame_hashes(
        self, video_path: str, frame_indices: list[int], resize_factor: float = 0.125
    ) -> dict[int, np.ndarray]:
        """Pre-compute masked perceptual hashes for frames in parallel."""
        if (
            not self.use_perceptual_hash
            or self.hasher is None
            or self._current_mask is None
        ):
            return {}

        # This cache should store pHash np.ndarrays
        # Check cache first - this assumes self.hash_cache[video_path] stores dict[int, np.ndarray]
        if video_path in self.hash_cache and all(
            idx in self.hash_cache[video_path] for idx in frame_indices
        ):
            return {
                idx: self.hash_cache[video_path][idx]
                for idx in frame_indices
                if idx in self.hash_cache[video_path]
            }

        logger.debug(f"Pre-computing masked hashes for {len(frame_indices)} frames")
        start_time = time.time()

        # Extract frames
        frames = self.processor.extract_frames(video_path, frame_indices, resize_factor)

        # Compute masked hashes in parallel
        hashes: dict[int, np.ndarray] = {}
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            future_to_idx = {
                executor.submit(
                    self._compute_masked_frame_hash, frame, self._current_mask
                ): idx
                for idx, frame in zip(frame_indices, frames, strict=False)
                if frame is not None
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    hash_value = future.result()  # This should be np.ndarray (pHash)
                    if hash_value is not None:
                        hashes[idx] = hash_value
                except Exception as e:
                    logger.warning(
                        f"Failed to compute masked hash for frame {idx}: {e}"
                    )

        # Update cache with pHash np.ndarrays
        if video_path not in self.hash_cache:
            self.hash_cache[video_path] = {}
        self.hash_cache[video_path].update(hashes)

        elapsed = time.time() - start_time
        logger.debug(f"Computed {len(hashes)} masked hashes in {elapsed:.2f}s")

        return hashes

    def _compute_masked_frame_hash(
        self, frame: np.ndarray, mask: np.ndarray
    ) -> np.ndarray | None:
        """Compute perceptual hash for masked frame."""
        if frame is None or self.hasher is None:
            return None

        # Resize mask to match frame if needed
        if frame.shape[:2] != mask.shape[:2]:
            resized_mask = cv2.resize(
                mask.astype(np.float32),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            resized_mask = (resized_mask > 0.5).astype(np.uint8)
        else:
            resized_mask = mask

        # Apply mask to frame
        if len(frame.shape) == 3:
            masked = frame.copy()
            for c in range(frame.shape[2]):
                masked[:, :, c] = frame[:, :, c] * resized_mask
        else:
            masked = frame * resized_mask

        # Crop to bounding box of mask
        rows = np.any(resized_mask, axis=1)
        cols = np.any(resized_mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            # Empty mask, compute hash on original
            resized = cv2.resize(frame, (32, 32))
        else:
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            cropped = masked[rmin : rmax + 1, cmin : cmax + 1]

            # Resize for hashing
            resized = cv2.resize(cropped, (32, 32))

        # Convert to grayscale if needed
        if len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        hash_value = self.hasher.compute(resized)  # This is the pHash np.ndarray
        return hash_value

    def _compute_frame_hash(self, frame: np.ndarray) -> np.ndarray | None:
        """Compute perceptual hash of a frame (specifically pHash)."""
        if frame is None or self.hasher is None:  # Check if hasher is initialized
            return None

        # Resize to standard size for consistent hashing
        resized = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)

        # Convert to grayscale if needed
        if len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Compute hash
        hash_value = self.hasher.compute(resized)  # This is the pHash np.ndarray
        return hash_value

    def _build_cost_matrix(
        self,
        bg_info: VideoInfo,
        fg_info: VideoInfo,
        bg_indices: list[int],
        fg_indices: list[int],
    ) -> np.ndarray | None:
        """Build cost matrix for dynamic programming alignment.

        Lower cost = better match. Uses perceptual hashes if available.
        """
        n_fg = len(fg_indices)
        n_bg = len(bg_indices)

        # Initialize cost matrix
        cost_matrix = np.full((n_fg, n_bg), np.inf)

        # Use hashes if available
        if (
            self.use_perceptual_hash
            and self.hasher is not None
            and bg_info.path in self.hash_cache
            and fg_info.path in self.hash_cache
        ):
            logger.debug("Building cost matrix using perceptual hashes")

            for i, fg_idx in enumerate(fg_indices):
                fg_hash = self.hash_cache[fg_info.path].get(fg_idx)
                if fg_hash is None:
                    continue

                for j, bg_idx in enumerate(bg_indices):
                    bg_hash = self.hash_cache[bg_info.path].get(bg_idx)
                    if bg_hash is None:
                        continue

                    # Hamming distance as cost
                    distance = cv2.norm(fg_hash, bg_hash, cv2.NORM_HAMMING)
                    cost_matrix[i, j] = distance
        else:
            logger.debug("Building cost matrix using frame extraction")
            # Parallelize frame extraction and comparison
            resize_factor = 0.25

            # Pre-extract all frames in parallel batches
            logger.info("Pre-extracting frames for cost matrix...")

            with ThreadPoolExecutor(max_workers=4) as executor:
                fg_future = executor.submit(
                    self.processor.extract_frames,
                    fg_info.path,
                    fg_indices,
                    resize_factor,
                )
                bg_future = executor.submit(
                    self.processor.extract_frames,
                    bg_info.path,
                    bg_indices,
                    resize_factor,
                )

                fg_frames = fg_future.result()
                bg_frames = bg_future.result()

            # Create frame dictionaries for fast lookup
            fg_frame_dict = {
                idx: frame
                for idx, frame in zip(fg_indices, fg_frames, strict=False)
                if frame is not None
            }
            bg_frame_dict = {
                idx: frame
                for idx, frame in zip(bg_indices, bg_frames, strict=False)
                if frame is not None
            }

            def compute_cell(i, j):
                """Compute single cell of cost matrix."""
                fg_idx = fg_indices[i]
                bg_idx = bg_indices[j]

                if fg_idx not in fg_frame_dict or bg_idx not in bg_frame_dict:
                    return i, j, np.inf

                fg_frame = fg_frame_dict[fg_idx]
                bg_frame = bg_frame_dict[bg_idx]

                # Apply mask if in border mode
                if self._current_mask is not None:
                    similarity = self._compute_frame_similarity(
                        bg_frame, fg_frame, self._current_mask
                    )
                else:
                    similarity = self._compute_frame_similarity(bg_frame, fg_frame)

                cost = 1.0 - similarity

                # Add temporal consistency penalty
                expected_j = int(i * n_bg / n_fg)
                time_penalty = 0.1 * abs(j - expected_j) / n_bg
                cost += time_penalty

                return i, j, cost

            # Process comparisons in parallel
            logger.info(f"Computing {n_fg * n_bg} similarities in parallel...")

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(
                    "  Computing similarities...", total=n_fg * n_bg
                )

                with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
                    # Submit all tasks
                    futures = []
                    for i in range(n_fg):
                        for j in range(n_bg):
                            future = executor.submit(compute_cell, i, j)
                            futures.append(future)

                    # Collect results
                    for future in as_completed(futures):
                        i, j, cost = future.result()
                        cost_matrix[i, j] = cost
                        progress.update(task, advance=1)

        return cost_matrix

    def _find_optimal_path(
        self, cost_matrix: np.ndarray, bg_indices: list[int], fg_indices: list[int]
    ) -> list[tuple[int, int, float]]:
        """Find optimal monotonic path through cost matrix using dynamic programming."""
        n_fg, n_bg = cost_matrix.shape

        # Dynamic programming table
        dp = np.full_like(cost_matrix, np.inf)
        parent = np.zeros_like(cost_matrix, dtype=int)

        # Initialize first row - first fg frame can match any bg frame
        dp[0, :] = cost_matrix[0, :]

        # Fill DP table
        for i in range(1, n_fg):
            for j in range(n_bg):
                # Can only come from previous bg frames (monotonic constraint)
                for k in range(j + 1):
                    if dp[i - 1, k] + cost_matrix[i, j] < dp[i, j]:
                        dp[i, j] = dp[i - 1, k] + cost_matrix[i, j]
                        parent[i, j] = k

        # Find best path by backtracking from minimum cost in last row
        min_j = np.argmin(dp[-1, :])
        path = []

        # Backtrack
        j = min_j
        for i in range(n_fg - 1, -1, -1):
            # Convert cost back to similarity
            similarity = 1.0 - cost_matrix[i, j]
            path.append((bg_indices[j], fg_indices[i], similarity))

            if i > 0:
                j = parent[i, j]

        path.reverse()

        # Filter out low-quality matches
        filtered_path = [
            match
            for match in path
            if match[2] > 0.5  # Similarity threshold
        ]

        return filtered_path

    def _refine_matches(
        self,
        initial_matches: list[tuple[int, int, float]],
        bg_info: VideoInfo,
        fg_info: VideoInfo,
        bg_indices: list[int],
        fg_indices: list[int],
    ) -> list[tuple[int, int, float]]:
        """Refine matches by adding intermediate keyframes where needed."""
        if len(initial_matches) < 2:
            return initial_matches

        refined = [initial_matches[0]]

        for i in range(1, len(initial_matches)):
            prev_match = refined[-1]
            curr_match = initial_matches[i]

            # Check if there's a large gap
            fg_gap = curr_match[1] - prev_match[1]
            curr_match[0] - prev_match[0]

            if fg_gap > 50:  # Large gap in foreground frames
                # Add intermediate keyframe
                mid_fg = (prev_match[1] + curr_match[1]) // 2
                mid_bg = (prev_match[0] + curr_match[0]) // 2

                # Find closest sampled indices
                closest_fg = min(fg_indices, key=lambda x: abs(x - mid_fg))
                closest_bg = min(bg_indices, key=lambda x: abs(x - mid_bg))

                # Compute similarity for intermediate frame
                if (
                    self.use_perceptual_hash
                    and fg_info.path in self.hash_cache
                    and bg_info.path in self.hash_cache
                ):
                    fg_hash = self.hash_cache[fg_info.path].get(closest_fg)
                    bg_hash = self.hash_cache[bg_info.path].get(closest_bg)

                    if fg_hash is not None and bg_hash is not None:
                        distance = cv2.norm(fg_hash, bg_hash, cv2.NORM_HAMMING)
                        similarity = 1.0 - (distance / 64.0)

                        if similarity > 0.5:
                            refined.append((closest_bg, closest_fg, similarity))

            refined.append(curr_match)

        return refined

    def _filter_monotonic(
        self, matches: list[tuple[int, int, float]]
    ) -> list[tuple[int, int, float]]:
        """Filter matches to ensure monotonic progression.

        This is now only used as a safety check since the DP algorithm
        already ensures monotonicity.
        """
        if not matches:
            return matches

        # Sort by foreground index
        matches.sort(key=lambda x: x[1])

        # Verify monotonicity (should already be monotonic from DP)
        filtered = []
        last_bg_idx = -1

        for bg_idx, fg_idx, sim in matches:
            if bg_idx > last_bg_idx:
                filtered.append((bg_idx, fg_idx, sim))
                last_bg_idx = bg_idx
            else:
                logger.warning(
                    f"Unexpected non-monotonic match: bg[{bg_idx}] for fg[{fg_idx}]"
                )

        return filtered

    def _build_frame_alignments(
        self,
        bg_info: VideoInfo,
        fg_info: VideoInfo,
        keyframe_matches: list[tuple[int, int, float]],
        trim: bool,
    ) -> list[FrameAlignment]:
        """Build complete frame-to-frame alignment.

        Creates alignment for EVERY foreground frame, finding the optimal
        background frame based on keyframe matches.
        """
        alignments = []

        # Determine range of foreground frames to process
        if trim and keyframe_matches:
            start_fg = keyframe_matches[0][1]
            end_fg = keyframe_matches[-1][1] + 1
        else:
            start_fg = 0
            end_fg = fg_info.frame_count

        logger.info(f"Building alignment for FG frames {start_fg} to {end_fg - 1}")

        # For each foreground frame, find optimal background frame
        for fg_idx in range(start_fg, end_fg):
            bg_idx = self._interpolate_bg_frame(
                fg_idx, keyframe_matches, bg_info, fg_info
            )

            # Ensure bg_idx is valid
            bg_idx = max(0, min(bg_idx, bg_info.frame_count - 1))

            # Estimate similarity based on nearby keyframes
            similarity = self._estimate_similarity(fg_idx, keyframe_matches)

            alignments.append(
                FrameAlignment(
                    fg_frame_idx=fg_idx,
                    bg_frame_idx=bg_idx,
                    similarity_score=similarity,
                )
            )

        logger.info(f"Created {len(alignments)} frame alignments")
        return alignments

    def _interpolate_bg_frame(
        self,
        fg_idx: int,
        keyframe_matches: list[tuple[int, int, float]],
        bg_info: VideoInfo,
        fg_info: VideoInfo,
    ) -> int:
        """Interpolate background frame index for given foreground frame.

        Uses smooth interpolation between keyframe matches to avoid
        sudden jumps or speed changes.
        """
        if not keyframe_matches:
            # Simple ratio-based mapping
            ratio = bg_info.fps / fg_info.fps if fg_info.fps > 0 else 1.0
            return int(fg_idx * ratio)

        # Find surrounding keyframes
        prev_match = None
        next_match = None

        for match in keyframe_matches:
            if match[1] <= fg_idx:
                prev_match = match
            elif match[1] > fg_idx and next_match is None:
                next_match = match
                break

        # Handle edge cases
        if prev_match is None:
            prev_match = keyframe_matches[0]
        if next_match is None:
            next_match = keyframe_matches[-1]

        # If at or beyond edges, extrapolate
        if fg_idx <= prev_match[1]:
            # Before first keyframe
            fps_ratio = bg_info.fps / fg_info.fps if fg_info.fps > 0 else 1.0
            offset = (prev_match[1] - fg_idx) * fps_ratio
            return int(prev_match[0] - offset)

        if fg_idx >= next_match[1]:
            # After last keyframe
            fps_ratio = bg_info.fps / fg_info.fps if fg_info.fps > 0 else 1.0
            offset = (fg_idx - next_match[1]) * fps_ratio
            return int(next_match[0] + offset)

        # Interpolate between keyframes
        if prev_match[1] == next_match[1]:
            return prev_match[0]

        # Calculate position ratio with smooth interpolation
        ratio = (fg_idx - prev_match[1]) / (next_match[1] - prev_match[1])

        # Apply smoothstep for more natural motion
        smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)

        # Interpolate background frame
        bg_idx = prev_match[0] + smooth_ratio * (next_match[0] - prev_match[0])

        return int(bg_idx)

    def _estimate_similarity(
        self, fg_idx: int, keyframe_matches: list[tuple[int, int, float]]
    ) -> float:
        """Estimate similarity score for a frame based on nearby keyframes."""
        if not keyframe_matches:
            return 0.5

        # Find closest keyframe
        min_dist = float("inf")
        closest_sim = 0.5

        for _, kf_fg_idx, similarity in keyframe_matches:
            dist = abs(fg_idx - kf_fg_idx)
            if dist < min_dist:
                min_dist = dist
                closest_sim = similarity

        # Decay similarity based on distance
        decay_rate = 0.95**min_dist
        return closest_sim * decay_rate

    def _calculate_alignment_confidence(
        self, keyframe_matches: list[tuple[int, int, float]]
    ) -> float:
        """Calculate overall confidence in the alignment."""
        if not keyframe_matches:
            return 0.0

        # Average similarity of matches
        avg_similarity = sum(m[2] for m in keyframe_matches) / len(keyframe_matches)

        # Coverage (how well distributed the matches are)
        coverage = len(keyframe_matches) / max(len(keyframe_matches), 20)

        return min(1.0, avg_similarity * coverage)

    def _create_direct_mapping(
        self, bg_info: VideoInfo, fg_info: VideoInfo
    ) -> TemporalAlignment:
        """Create simple direct frame mapping as fallback."""
        fps_ratio = bg_info.fps / fg_info.fps if fg_info.fps > 0 else 1.0

        alignments = []
        for fg_idx in range(fg_info.frame_count):
            bg_idx = int(fg_idx * fps_ratio)
            bg_idx = min(bg_idx, bg_info.frame_count - 1)

            alignments.append(
                FrameAlignment(
                    fg_frame_idx=fg_idx, bg_frame_idx=bg_idx, similarity_score=0.5
                )
            )

        return TemporalAlignment(
            offset_seconds=0.0,
            frame_alignments=alignments,
            method_used="direct",
            confidence=0.3,
        )

    def _align_frames_dtw(
        self, bg_info: VideoInfo, fg_info: VideoInfo, trim: bool = False
    ) -> TemporalAlignment:
        """Align videos using Dynamic Time Warping for guaranteed monotonic alignment.

        This is the new improved method that replaces the problematic keyframe matching.

        Why DTW:
        - Guarantees monotonic alignment (no backward jumps)
        - Finds globally optimal path
        - Handles speed variations naturally
        - No more drift or catch-up issues
        """
        logger.info("Using DTW-based temporal alignment")

        # Initialize fingerprinter if needed
        if self.fingerprinter is None:
            try:
                self.fingerprinter = FrameFingerprinter()
            except Exception as e:
                logger.error(f"Failed to initialize fingerprinter: {e}")
                logger.warning("Falling back to classic alignment")
                self.use_dtw = False
                return self.align_frames(bg_info, fg_info, trim)

        # Determine frames to sample
        fg_sample_interval = max(1, fg_info.frame_count // self.max_keyframes)
        bg_sample_interval = max(1, bg_info.frame_count // (self.max_keyframes * 2))

        fg_indices = list(range(0, fg_info.frame_count, fg_sample_interval))
        bg_indices = list(range(0, bg_info.frame_count, bg_sample_interval))

        # Always include first and last frames
        if 0 not in fg_indices:
            fg_indices.insert(0, 0)
        if fg_info.frame_count - 1 not in fg_indices:
            fg_indices.append(fg_info.frame_count - 1)
        if 0 not in bg_indices:
            bg_indices.insert(0, 0)
        if bg_info.frame_count - 1 not in bg_indices:
            bg_indices.append(bg_info.frame_count - 1)

        logger.info(
            f"DTW sampling: {len(fg_indices)} FG frames, {len(bg_indices)} BG frames"
        )

        # Compute fingerprints for sampled frames
        logger.info("Computing frame fingerprints...")

        # Check if we need to use masked fingerprints
        if self._current_mask is not None:
            logger.info("Using masked fingerprints for border mode DTW alignment")
            fg_fingerprints = self._precompute_masked_video_fingerprints(
                fg_info.path, fg_indices, self._current_mask, resize_factor=0.25
            )
            bg_fingerprints = self._precompute_masked_video_fingerprints(
                bg_info.path, bg_indices, self._current_mask, resize_factor=0.25
            )
        else:
            fg_fingerprints = self.fingerprinter.precompute_video_fingerprints(
                fg_info.path, fg_indices, self.processor, resize_factor=0.25
            )
            bg_fingerprints = self.fingerprinter.precompute_video_fingerprints(
                bg_info.path, bg_indices, self.processor, resize_factor=0.25
            )

        if not fg_fingerprints or not bg_fingerprints:
            logger.error("Failed to compute fingerprints")
            logger.warning("Falling back to classic alignment")
            self.use_dtw = False
            return self.align_frames(bg_info, fg_info, trim)

        # Run DTW alignment
        logger.info("Running DTW alignment...")
        dtw_matches = self.dtw_aligner.align_videos(
            fg_fingerprints,
            bg_fingerprints,
            self.fingerprinter.compare_fingerprints,
            show_progress=True,
        )

        if not dtw_matches:
            logger.warning("DTW produced no matches, using direct mapping")
            return self._create_direct_mapping(bg_info, fg_info)

        # Create complete frame alignments from DTW matches
        frame_alignments = self.dtw_aligner.create_frame_alignments(
            dtw_matches, fg_info.frame_count, bg_info.frame_count
        )

        # Apply trimming if requested
        if trim and frame_alignments:
            # Find actual matched range
            bg_indices_used = [a.bg_frame_idx for a in frame_alignments]
            min(bg_indices_used)
            max(bg_indices_used)

            # Trim to only include frames with good matches
            trimmed_alignments = []
            for alignment in frame_alignments:
                if alignment.similarity_score > 0.5:  # Quality threshold
                    trimmed_alignments.append(alignment)

            if trimmed_alignments:
                frame_alignments = trimmed_alignments

        # Calculate overall temporal offset
        if dtw_matches:
            first_match = dtw_matches[0]
            offset_seconds = (first_match[0] / bg_info.fps) - (
                first_match[1] / fg_info.fps
            )
        else:
            offset_seconds = 0.0

        # Calculate overall confidence
        if frame_alignments:
            avg_confidence = sum(a.similarity_score for a in frame_alignments) / len(
                frame_alignments
            )
        else:
            avg_confidence = 0.5

        logger.info(
            f"DTW alignment complete: {len(frame_alignments)} frames, "
            f"offset={offset_seconds:.3f}s, confidence={avg_confidence:.3f}"
        )

        return TemporalAlignment(
            offset_seconds=offset_seconds,
            frame_alignments=frame_alignments,
            method_used="dtw",
            confidence=avg_confidence,
        )

    def create_border_mask(
        self,
        spatial_alignment,
        fg_info: VideoInfo,
        bg_info: VideoInfo,
        border_thickness: int = 8,
    ) -> np.ndarray:
        """Create border mask for border-based temporal alignment.

        The border mask defines the region around the foreground video edges where
        background video is visible. This is used for similarity comparison in border mode.

        Args:
            spatial_alignment: Result from spatial alignment containing x/y offsets
            fg_info: Foreground video information
            bg_info: Background video information
            border_thickness: Thickness of border region in pixels

        Returns:
            Binary mask where 1 indicates border region, 0 indicates non-border
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

    def align_frames_with_mask(
        self,
        bg_info: VideoInfo,
        fg_info: VideoInfo,
        trim: bool = False,
        mask: np.ndarray | None = None,
    ) -> TemporalAlignment:
        self._current_mask = mask

        original_use_dtw = self.use_dtw
        original_use_perceptual_hash = self.use_perceptual_hash

        effective_use_dtw = self.use_dtw
        effective_use_perceptual_hash = self.use_perceptual_hash

        if mask is not None:
            logger.info("Border mode active for temporal alignment.")
            if self.use_dtw:
                logger.info(
                    "Border mode with DTW enabled using masked perceptual hashing."
                )
                # DTW now supports masked perceptual hashing
                effective_use_dtw = True

            # In border mode, we can now use masked perceptual hashing
            if not effective_use_dtw and self.use_perceptual_hash:
                logger.info(
                    "Border mode: masked perceptual hashing for faster processing."
                )
                effective_use_perceptual_hash = True
            elif not effective_use_dtw and not self.use_perceptual_hash:
                logger.info(
                    "Border mode: SSIM comparison (perceptual hash unavailable)."
                )
                effective_use_perceptual_hash = False

        alignment_result: TemporalAlignment
        try:
            # Attempt to initialize fingerprinter for DTW if needed and not already done
            if effective_use_dtw and self.fingerprinter is None:
                try:
                    self.fingerprinter = FrameFingerprinter()
                except Exception as e:
                    logger.error(f"Failed to init FrameFingerprinter for DTW: {e}")
                    logger.warning("Falling back to classic alignment.")
                    effective_use_dtw = False
                    self.use_perceptual_hash = original_use_perceptual_hash
                    effective_use_perceptual_hash = original_use_perceptual_hash

            if effective_use_dtw and self.fingerprinter is not None:
                logger.info("Using DTW-based temporal alignment (full frame).")
                alignment_result = self._align_frames_dtw(bg_info, fg_info, trim)
            else:
                log_hash_status = (
                    "Enabled"
                    if self.use_perceptual_hash and self.hasher
                    else "Disabled (SSIM)"
                )
                logger.info(
                    f"Using classic (keyframe-based) temporal alignment. "
                    f"Hash/SSIM: {log_hash_status}."
                )
                keyframe_matches = self._find_keyframe_matches(bg_info, fg_info)

                if not keyframe_matches:
                    logger.warning("No keyframe matches found, using direct mapping.")
                    alignment_result = self._create_direct_mapping(bg_info, fg_info)
                else:
                    frame_alignments = self._build_frame_alignments(
                        bg_info, fg_info, keyframe_matches, trim
                    )
                    first_match = keyframe_matches[0]
                    offset_seconds = (first_match[0] / bg_info.fps) - (
                        first_match[1] / fg_info.fps
                    )
                    confidence = self._calculate_alignment_confidence(keyframe_matches)

                    method_detail = (
                        "hash"
                        if effective_use_perceptual_hash and self.hasher
                        else "SSIM"
                    )
                    base_method_str = "border" if mask is not None else "frames"
                    method_used_str = f"{base_method_str} (classic/{method_detail})"

                    alignment_result = TemporalAlignment(
                        offset_seconds=offset_seconds,
                        frame_alignments=frame_alignments,
                        method_used=method_used_str,
                        confidence=confidence,
                    )
            return alignment_result
        finally:
            self._current_mask = None
            self.use_dtw = original_use_dtw
            self.use_perceptual_hash = original_use_perceptual_hash

    def _find_keyframe_matches_with_mask(
        self, bg_info: VideoInfo, fg_info: VideoInfo, mask: np.ndarray | None = None
    ) -> list[tuple[int, int, float]]:
        """Find matching keyframes between videos with optional mask support."""
        # This is similar to _find_keyframe_matches but uses masked similarity
        # For now, we'll modify the existing method to support masks
        return self._find_keyframe_matches(bg_info, fg_info)
