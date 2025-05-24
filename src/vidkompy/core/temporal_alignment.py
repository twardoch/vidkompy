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
from typing import List, Tuple, Optional
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console

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
    
    def _find_keyframe_matches(
        self,
        bg_info: VideoInfo,
        fg_info: VideoInfo
    ) -> List[Tuple[int, int, float]]:
        """Find matching keyframes between videos.
        
        Returns:
            List of (bg_frame_idx, fg_frame_idx, similarity) tuples
        """
        # Determine sampling rate
        target_keyframes = min(self.max_keyframes, fg_info.frame_count)
        sample_interval = max(1, fg_info.frame_count // target_keyframes)
        
        logger.info(f"Sampling every {sample_interval} frames for keyframe matching")
        
        # Extract sampled frames
        fg_indices = list(range(0, fg_info.frame_count, sample_interval))
        fg_frames = self.processor.extract_frames(
            fg_info.path, fg_indices, resize_factor=0.25
        )
        
        # Sample background more densely for better matching
        bg_sample_interval = max(1, sample_interval // 2)
        bg_indices = list(range(0, bg_info.frame_count, bg_sample_interval))
        bg_frames = self.processor.extract_frames(
            bg_info.path, bg_indices, resize_factor=0.25
        )
        
        if not fg_frames or not bg_frames:
            logger.error("Failed to extract frames for matching")
            return []
        
        # Find matches with progress
        matches = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(
                "Matching keyframes...", 
                total=len(fg_frames)
            )
            
            for fg_idx, fg_frame in enumerate(fg_frames):
                if fg_frame is None:
                    progress.update(task, advance=1)
                    continue
                    
                fg_actual_idx = fg_indices[fg_idx]
                best_match = self._find_best_match(
                    fg_frame, fg_actual_idx, bg_frames, bg_indices, 
                    fg_info, bg_info
                )
                
                if best_match and best_match[2] > 0.6:  # Similarity threshold
                    matches.append(best_match)
                
                progress.update(task, advance=1)
        
        # Filter to ensure monotonic progression
        matches = self._filter_monotonic(matches)
        
        logger.info(f"Found {len(matches)} keyframe matches")
        return matches
    
    def _find_best_match(
        self,
        fg_frame: np.ndarray,
        fg_idx: int,
        bg_frames: List[np.ndarray],
        bg_indices: List[int],
        fg_info: VideoInfo,
        bg_info: VideoInfo
    ) -> Optional[Tuple[int, int, float]]:
        """Find best matching background frame for a foreground frame.
        
        Returns:
            (bg_frame_idx, fg_frame_idx, similarity) or None
        """
        # Estimate expected position
        time_ratio = bg_info.duration / fg_info.duration if fg_info.duration > 0 else 1.0
        expected_bg_idx = int(fg_idx * time_ratio * (bg_info.fps / fg_info.fps))
        
        # Search window (adaptive based on confidence)
        window_size = max(len(bg_frames) // 10, 20)  # At least 20 frames
        
        best_similarity = 0.0
        best_bg_idx = -1
        
        for i, bg_frame in enumerate(bg_frames):
            if bg_frame is None:
                continue
                
            actual_bg_idx = bg_indices[i]
            
            # Skip if too far from expected position (unless early in video)
            if fg_idx > 50 and abs(actual_bg_idx - expected_bg_idx) > window_size * 2:
                continue
            
            similarity = self._compute_frame_similarity(bg_frame, fg_frame)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_bg_idx = actual_bg_idx
        
        if best_bg_idx >= 0:
            return (best_bg_idx, fg_idx, best_similarity)
        return None
    
    def _compute_frame_similarity(
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
    
    def _filter_monotonic(
        self,
        matches: List[Tuple[int, int, float]]
    ) -> List[Tuple[int, int, float]]:
        """Filter matches to ensure monotonic progression."""
        if not matches:
            return matches
        
        # Sort by foreground index
        matches.sort(key=lambda x: x[1])
        
        # Filter non-monotonic background indices
        filtered = []
        last_bg_idx = -1
        
        for bg_idx, fg_idx, sim in matches:
            if bg_idx > last_bg_idx:
                filtered.append((bg_idx, fg_idx, sim))
                last_bg_idx = bg_idx
            else:
                logger.debug(
                    f"Filtering non-monotonic match: bg[{bg_idx}] for fg[{fg_idx}]"
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