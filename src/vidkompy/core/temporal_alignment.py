#!/usr/bin/env python3
# this_file: src/vidkompy/core/temporal_alignment.py

"""
Temporal alignment module for synchronizing videos.

Implements audio-based and frame-based temporal alignment with
emphasis on preserving all foreground frames without retiming.
"""

import cv2
import numpy as np
import soundfile as sf
from scipy import signal
from skimage.metrics import structural_similarity as ssim
from typing import List, Tuple, Optional, Dict
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from ..models import VideoInfo, FrameAlignment, TemporalAlignment
from .video_processor import VideoProcessor

console = Console()


class TemporalAligner:
    """Handles temporal alignment between videos."""
    
    def __init__(self, processor: VideoProcessor, max_keyframes: int = 2000):
        """Initialize temporal aligner.
        
        Args:
            processor: Video processor instance
            max_keyframes: Maximum keyframes for frame matching
        """
        self.processor = processor
        self.max_keyframes = max_keyframes
        self.use_perceptual_hash = True  # Enable by default
        self.hash_cache: Dict[str, Dict[int, np.ndarray]] = {}
        
        # Try to initialize perceptual hasher
        try:
            self.hasher = cv2.img_hash.PHash_create()
            logger.info("✓ Perceptual hashing enabled (pHash)")
        except AttributeError:
            logger.warning("opencv-contrib-python not installed, falling back to SSIM")
            self.use_perceptual_hash = False
            self.hasher = None
    
    def align_audio(
        self,
        bg_audio_path: str,
        fg_audio_path: str
    ) -> float:
        """Compute temporal offset using audio cross-correlation.
        
        Args:
            bg_audio_path: Background audio WAV file
            fg_audio_path: Foreground audio WAV file
            
        Returns:
            Offset in seconds (positive means FG starts later)
        """
        logger.debug("Computing audio cross-correlation")
        
        # Load audio
        bg_audio, bg_sr = sf.read(bg_audio_path)
        fg_audio, fg_sr = sf.read(fg_audio_path)
        
        if bg_sr != fg_sr:
            logger.warning(f"Sample rates differ: {bg_sr} vs {fg_sr}")
            return 0.0
        
        # Compute cross-correlation
        correlation = signal.correlate(bg_audio, fg_audio, mode='full', method='fft')
        
        # Find peak
        peak_idx = np.argmax(np.abs(correlation))
        
        # Convert to time offset
        center = len(bg_audio) - 1
        lag_samples = peak_idx - center
        offset_seconds = lag_samples / bg_sr
        
        # Calculate confidence
        peak_value = np.abs(correlation[peak_idx])
        avg_value = np.mean(np.abs(correlation))
        confidence = peak_value / avg_value if avg_value > 0 else 0
        
        logger.info(
            f"Audio alignment: offset={offset_seconds:.3f}s, "
            f"confidence={confidence:.2f}"
        )
        
        return offset_seconds
    
    def align_frames(
        self,
        bg_info: VideoInfo,
        fg_info: VideoInfo,
        trim: bool = False
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
        
        # Sample frames for keyframe matching
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
            confidence=self._calculate_alignment_confidence(keyframe_matches)
        )
    
    def _precompute_frame_hashes(
        self,
        video_path: str,
        frame_indices: List[int],
        resize_factor: float = 0.125
    ) -> Dict[int, np.ndarray]:
        """Pre-compute perceptual hashes for frames in parallel."""
        if not self.use_perceptual_hash or self.hasher is None:
            return {}
        
        # Check cache first
        if video_path in self.hash_cache:
            cached = self.hash_cache[video_path]
            if all(idx in cached for idx in frame_indices):
                return {idx: cached[idx] for idx in frame_indices}
        
        logger.debug(f"Pre-computing hashes for {len(frame_indices)} frames")
        start_time = time.time()
        
        # Extract frames
        frames = self.processor.extract_frames(video_path, frame_indices, resize_factor)
        
        # Compute hashes in parallel
        hashes = {}
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            future_to_idx = {
                executor.submit(self._compute_frame_hash, frame): idx
                for idx, frame in zip(frame_indices, frames)
                if frame is not None
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    hash_value = future.result()
                    hashes[idx] = hash_value
                except Exception as e:
                    logger.warning(f"Failed to compute hash for frame {idx}: {e}")
        
        # Update cache
        if video_path not in self.hash_cache:
            self.hash_cache[video_path] = {}
        self.hash_cache[video_path].update(hashes)
        
        elapsed = time.time() - start_time
        logger.debug(f"Computed {len(hashes)} hashes in {elapsed:.2f}s")
        
        return hashes

    def _find_keyframe_matches(
        self,
        bg_info: VideoInfo,
        fg_info: VideoInfo
    ) -> List[Tuple[int, int, float]]:
        """Find matching keyframes between videos using monotonic dynamic programming.
        
        Returns:
            List of (bg_frame_idx, fg_frame_idx, similarity) tuples
        """
        # Determine sampling rate
        target_keyframes = min(self.max_keyframes, fg_info.frame_count)
        sample_interval = max(1, fg_info.frame_count // target_keyframes)
        
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
        
        logger.info(f"Sampling {len(fg_indices)} FG frames and {len(bg_indices)} BG frames")
        
        # Pre-compute all hashes if available
        if self.use_perceptual_hash and self.hasher is not None:
            logger.info("Pre-computing perceptual hashes...")
            fg_hashes = self._precompute_frame_hashes(fg_info.path, fg_indices, 0.125)
            bg_hashes = self._precompute_frame_hashes(bg_info.path, bg_indices, 0.125)
            
            if not fg_hashes or not bg_hashes:
                logger.error("Failed to compute hashes, falling back to SSIM")
                self.use_perceptual_hash = False
        
        # Build cost matrix using dynamic programming approach
        logger.info("Building cost matrix for dynamic programming alignment...")
        cost_matrix = self._build_cost_matrix(
            bg_info, fg_info, bg_indices, fg_indices
        )
        
        if cost_matrix is None:
            logger.error("Failed to build cost matrix")
            return []
        
        # Find optimal monotonic path through cost matrix
        matches = self._find_optimal_path(
            cost_matrix, bg_indices, fg_indices
        )
        
        # Validate matches with higher quality check if needed
        if len(matches) < 10:
            logger.warning(f"Only found {len(matches)} matches, attempting refinement...")
            matches = self._refine_matches(
                matches, bg_info, fg_info, bg_indices, fg_indices
            )
        
        logger.info(f"Found {len(matches)} monotonic keyframe matches")
        
        if self.use_perceptual_hash:
            logger.info("✓ Perceptual hashing provided significant speedup")
        
        return matches
    
    
    def _compute_frame_similarity(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> float:
        """Compute similarity between two frames using perceptual hash or SSIM."""
        if self.use_perceptual_hash and self.hasher is not None:
            # Use perceptual hash for fast comparison
            hash1 = self._compute_frame_hash(frame1)
            hash2 = self._compute_frame_hash(frame2)
            
            # Compute Hamming distance
            distance = cv2.norm(hash1, hash2, cv2.NORM_HAMMING)
            
            # Convert to similarity score (0-1)
            # pHash produces 64-bit hash (8 bytes)
            max_distance = 64  # Maximum possible Hamming distance
            similarity = 1.0 - (distance / max_distance)
            
            return float(similarity)
        else:
            # Fallback to SSIM
            return self._compute_ssim_similarity(frame1, frame2)
    
    def _compute_ssim_similarity(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> float:
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
    
    def _compute_frame_hash(self, frame: np.ndarray) -> np.ndarray:
        """Compute perceptual hash of a frame."""
        # Resize to standard size for consistent hashing
        resized = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale if needed
        if len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Compute hash
        hash_value = self.hasher.compute(resized)
        return hash_value
    
    def _build_cost_matrix(
        self,
        bg_info: VideoInfo,
        fg_info: VideoInfo,
        bg_indices: List[int],
        fg_indices: List[int]
    ) -> Optional[np.ndarray]:
        """Build cost matrix for dynamic programming alignment.
        
        Lower cost = better match. Uses perceptual hashes if available.
        """
        n_fg = len(fg_indices)
        n_bg = len(bg_indices)
        
        # Initialize cost matrix
        cost_matrix = np.full((n_fg, n_bg), np.inf)
        
        # Use hashes if available
        if (self.use_perceptual_hash and self.hasher is not None and
            bg_info.path in self.hash_cache and fg_info.path in self.hash_cache):
            
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
            # Fallback to extracting and comparing frames
            resize_factor = 0.25
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task(
                    "Building cost matrix...", 
                    total=n_fg * n_bg
                )
                
                for i, fg_idx in enumerate(fg_indices):
                    fg_frames = self.processor.extract_frames(
                        fg_info.path, [fg_idx], resize_factor
                    )
                    if not fg_frames or fg_frames[0] is None:
                        progress.update(task, advance=n_bg)
                        continue
                    
                    fg_frame = fg_frames[0]
                    
                    for j, bg_idx in enumerate(bg_indices):
                        bg_frames = self.processor.extract_frames(
                            bg_info.path, [bg_idx], resize_factor
                        )
                        if not bg_frames or bg_frames[0] is None:
                            progress.update(task, advance=1)
                            continue
                        
                        bg_frame = bg_frames[0]
                        
                        # Use SSIM as similarity, convert to cost
                        similarity = self._compute_ssim_similarity(bg_frame, fg_frame)
                        cost_matrix[i, j] = 1.0 - similarity
                        
                        progress.update(task, advance=1)
        
        # Add temporal consistency penalty
        # Penalize large jumps in time
        for i in range(n_fg):
            for j in range(n_bg):
                # Expected position based on linear time mapping
                expected_j = int(i * n_bg / n_fg)
                time_penalty = 0.1 * abs(j - expected_j) / n_bg
                cost_matrix[i, j] += time_penalty
        
        return cost_matrix
    
    def _find_optimal_path(
        self,
        cost_matrix: np.ndarray,
        bg_indices: List[int],
        fg_indices: List[int]
    ) -> List[Tuple[int, int, float]]:
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
                    if dp[i-1, k] + cost_matrix[i, j] < dp[i, j]:
                        dp[i, j] = dp[i-1, k] + cost_matrix[i, j]
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
            match for match in path 
            if match[2] > 0.5  # Similarity threshold
        ]
        
        return filtered_path
    
    def _refine_matches(
        self,
        initial_matches: List[Tuple[int, int, float]],
        bg_info: VideoInfo,
        fg_info: VideoInfo,
        bg_indices: List[int],
        fg_indices: List[int]
    ) -> List[Tuple[int, int, float]]:
        """Refine matches by adding intermediate keyframes where needed."""
        if len(initial_matches) < 2:
            return initial_matches
        
        refined = [initial_matches[0]]
        
        for i in range(1, len(initial_matches)):
            prev_match = refined[-1]
            curr_match = initial_matches[i]
            
            # Check if there's a large gap
            fg_gap = curr_match[1] - prev_match[1]
            bg_gap = curr_match[0] - prev_match[0]
            
            if fg_gap > 50:  # Large gap in foreground frames
                # Add intermediate keyframe
                mid_fg = (prev_match[1] + curr_match[1]) // 2
                mid_bg = (prev_match[0] + curr_match[0]) // 2
                
                # Find closest sampled indices
                closest_fg = min(fg_indices, key=lambda x: abs(x - mid_fg))
                closest_bg = min(bg_indices, key=lambda x: abs(x - mid_bg))
                
                # Compute similarity for intermediate frame
                if (self.use_perceptual_hash and 
                    fg_info.path in self.hash_cache and 
                    bg_info.path in self.hash_cache):
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
        self,
        matches: List[Tuple[int, int, float]]
    ) -> List[Tuple[int, int, float]]:
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
        keyframe_matches: List[Tuple[int, int, float]],
        trim: bool
    ) -> List[FrameAlignment]:
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
        
        logger.info(f"Building alignment for FG frames {start_fg} to {end_fg-1}")
        
        # For each foreground frame, find optimal background frame
        for fg_idx in range(start_fg, end_fg):
            bg_idx = self._interpolate_bg_frame(
                fg_idx, keyframe_matches, bg_info, fg_info
            )
            
            # Ensure bg_idx is valid
            bg_idx = max(0, min(bg_idx, bg_info.frame_count - 1))
            
            # Estimate similarity based on nearby keyframes
            similarity = self._estimate_similarity(fg_idx, keyframe_matches)
            
            alignments.append(FrameAlignment(
                fg_frame_idx=fg_idx,
                bg_frame_idx=bg_idx,
                similarity_score=similarity
            ))
        
        logger.info(f"Created {len(alignments)} frame alignments")
        return alignments
    
    def _interpolate_bg_frame(
        self,
        fg_idx: int,
        keyframe_matches: List[Tuple[int, int, float]],
        bg_info: VideoInfo,
        fg_info: VideoInfo
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
        self,
        fg_idx: int,
        keyframe_matches: List[Tuple[int, int, float]]
    ) -> float:
        """Estimate similarity score for a frame based on nearby keyframes."""
        if not keyframe_matches:
            return 0.5
        
        # Find closest keyframe
        min_dist = float('inf')
        closest_sim = 0.5
        
        for _, kf_fg_idx, similarity in keyframe_matches:
            dist = abs(fg_idx - kf_fg_idx)
            if dist < min_dist:
                min_dist = dist
                closest_sim = similarity
        
        # Decay similarity based on distance
        decay_rate = 0.95 ** min_dist
        return closest_sim * decay_rate
    
    def _calculate_alignment_confidence(
        self,
        keyframe_matches: List[Tuple[int, int, float]]
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
        self,
        bg_info: VideoInfo,
        fg_info: VideoInfo
    ) -> TemporalAlignment:
        """Create simple direct frame mapping as fallback."""
        fps_ratio = bg_info.fps / fg_info.fps if fg_info.fps > 0 else 1.0
        
        alignments = []
        for fg_idx in range(fg_info.frame_count):
            bg_idx = int(fg_idx * fps_ratio)
            bg_idx = min(bg_idx, bg_info.frame_count - 1)
            
            alignments.append(FrameAlignment(
                fg_frame_idx=fg_idx,
                bg_frame_idx=bg_idx,
                similarity_score=0.5
            ))
        
        return TemporalAlignment(
            offset_seconds=0.0,
            frame_alignments=alignments,
            method_used="direct",
            confidence=0.3
        )