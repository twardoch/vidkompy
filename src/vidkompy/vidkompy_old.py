#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["fire", "rich", "loguru", "opencv-python", "numpy", "scipy", "ffmpeg-python", "soundfile", "scikit-image"]
# ///
# this_file: vidoverlay.py

"""
Video overlay tool with intelligent spatial and temporal alignment.

Overlays a foreground video onto a background video with automatic:
- Spatial alignment using template matching and feature detection
- Temporal alignment using audio cross-correlation
- Frame rate harmonization
- Duration management
"""

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
from enum import Enum

import cv2
import ffmpeg
import fire
import numpy as np
import soundfile as sf
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from scipy import signal
from skimage.metrics import structural_similarity as ssim

console = Console()


class AlignmentMode(Enum):
    """Temporal alignment methods."""

    AUDIO = "audio"
    DURATION = "duration"
    FRAMES = "frames"


@dataclass
class VideoInfo:
    """Video metadata container."""

    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool
    audio_sample_rate: int | None = None
    audio_channels: int | None = None
    path: str = ""
    frame_count: int = 0


@dataclass
class FrameAlignment:
    """Frame-to-frame alignment mapping."""

    bg_frame_idx: int
    fg_frame_idx: int
    similarity_score: float
    temporal_offset: float = 0.0


class VideoOverlay:
    """Main video overlay processor."""

    def __init__(self, verbose: bool = False):
        """Initialize the video overlay processor.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self._setup_logging()

    def _setup_logging(self):
        """Configure loguru logging."""
        logger.remove()  # Remove default handler

        if self.verbose:
            logger.add(
                sys.stderr,
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan> - <level>{message}</level>",
                level="DEBUG",
            )
        else:
            logger.add(
                sys.stderr,
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
                level="INFO",
            )

    def get_video_info(self, video_path: str) -> VideoInfo:
        """Extract video metadata using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            VideoInfo object with metadata
        """
        logger.debug(f"Probing video: {video_path}")

        try:
            # Get video stream info
            probe = ffmpeg.probe(video_path)

            video_stream = next(
                (s for s in probe["streams"] if s["codec_type"] == "video"), None
            )

            if not video_stream:
                raise ValueError(f"No video stream found in {video_path}")

            # Extract video properties
            width = int(video_stream["width"])
            height = int(video_stream["height"])

            # Parse frame rate (can be rational like "24000/1001")
            fps_str = video_stream.get("r_frame_rate", "0/1")
            if "/" in fps_str:
                num, den = map(int, fps_str.split("/"))
                fps = num / den if den != 0 else 0
            else:
                fps = float(fps_str)

            # Get duration
            duration = float(probe["format"].get("duration", 0))

            # Calculate frame count
            frame_count = int(video_stream.get("nb_frames", 0))
            if frame_count == 0 and duration > 0 and fps > 0:
                frame_count = int(duration * fps)

            # Check for audio
            audio_stream = next(
                (s for s in probe["streams"] if s["codec_type"] == "audio"), None
            )

            has_audio = audio_stream is not None
            audio_sample_rate = None
            audio_channels = None

            if has_audio and audio_stream:
                audio_sample_rate = int(audio_stream.get("sample_rate", 0))
                audio_channels = int(audio_stream.get("channels", 0))

            info = VideoInfo(
                width=width,
                height=height,
                fps=fps,
                duration=duration,
                has_audio=has_audio,
                audio_sample_rate=audio_sample_rate,
                audio_channels=audio_channels,
                path=video_path,
                frame_count=frame_count,
            )

            logger.info(
                f"Video info for {Path(video_path).name}: "
                f"{width}x{height}, {fps:.2f} fps, {duration:.2f}s, "
                f"{frame_count} frames, audio: {'yes' if has_audio else 'no'}"
            )

            return info

        except Exception as e:
            logger.error(f"Failed to probe video {video_path}: {e}")
            raise

    def extract_audio(
        self, video_path: str, output_path: str, sample_rate: int = 16000
    ) -> bool:
        """Extract audio from video to WAV file.

        Args:
            video_path: Input video path
            output_path: Output WAV path
            sample_rate: Target sample rate for audio

        Returns:
            True if extraction successful, False otherwise
        """
        logger.debug(f"Extracting audio from {video_path} to {output_path}")

        try:
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(
                stream,
                output_path,
                acodec="pcm_s16le",
                ac=1,  # Convert to mono
                ar=sample_rate,
                loglevel="error",
            )
            ffmpeg.run(stream, overwrite_output=True)
            return True

        except ffmpeg.Error as e:
            logger.error(f"Audio extraction failed: {e.stderr.decode()}")
            return False

    def compute_audio_offset(self, bg_audio_path: str, fg_audio_path: str) -> float:
        """Compute temporal offset using audio cross-correlation.

        Args:
            bg_audio_path: Background audio WAV file
            fg_audio_path: Foreground audio WAV file

        Returns:
            Offset in seconds (positive means FG starts later)
        """
        logger.debug("Computing audio cross-correlation")

        # Load audio files
        bg_audio, bg_sr = sf.read(bg_audio_path)
        fg_audio, fg_sr = sf.read(fg_audio_path)

        if bg_sr != fg_sr:
            logger.warning(f"Sample rates don't match: {bg_sr} vs {fg_sr}")
            return 0.0

        # Compute cross-correlation
        correlation = signal.correlate(bg_audio, fg_audio, mode="full", method="fft")

        # Find peak
        peak_idx = np.argmax(np.abs(correlation))

        # Convert to time offset
        # Center of correlation is at len(bg_audio) - 1
        center = len(bg_audio) - 1
        lag_samples = peak_idx - center
        offset_seconds = lag_samples / bg_sr

        # Get correlation strength
        peak_value = np.abs(correlation[peak_idx])
        avg_value = np.mean(np.abs(correlation))
        confidence = peak_value / avg_value if avg_value > 0 else 0

        logger.info(
            f"Audio alignment: offset={offset_seconds:.3f}s, "
            f"confidence={confidence:.2f}"
        )

        return offset_seconds

    def compute_spatial_offset(
        self, bg_frame: np.ndarray, fg_frame: np.ndarray, method: str = "precise"
    ) -> tuple[int, int]:
        """Compute spatial offset for overlay positioning.

        Args:
            bg_frame: Background video frame
            fg_frame: Foreground video frame
            method: Alignment method ('precise' or 'fast')

        Returns:
            (x, y) offset for foreground placement
        """
        bg_h, bg_w = bg_frame.shape[:2]
        fg_h, fg_w = fg_frame.shape[:2]

        # If foreground is larger, we'll need to scale it down
        if fg_w > bg_w or fg_h > bg_h:
            logger.warning("Foreground larger than background, will scale down")
            return 0, 0

        if method == "precise":
            return self._template_matching(bg_frame, fg_frame)
        elif method == "fast":
            return self._feature_matching(bg_frame, fg_frame)
        else:
            # Default to center
            return (bg_w - fg_w) // 2, (bg_h - fg_h) // 2

    def _template_matching(
        self, bg_frame: np.ndarray, fg_frame: np.ndarray
    ) -> tuple[int, int]:
        """Find best position using template matching.

        Args:
            bg_frame: Background frame
            fg_frame: Foreground frame (template)

        Returns:
            (x, y) position with best match
        """
        logger.debug("Using template matching for spatial alignment")

        # Convert to grayscale for matching
        bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
        fg_gray = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2GRAY)

        # Apply template matching
        result = cv2.matchTemplate(bg_gray, fg_gray, cv2.TM_CCOEFF_NORMED)

        # Find best match location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        logger.info(f"Template match confidence: {max_val:.3f}")

        # If confidence is too low, fall back to center
        if max_val < 0.7:
            logger.warning("Low template match confidence, defaulting to center")
            bg_h, bg_w = bg_frame.shape[:2]
            fg_h, fg_w = fg_frame.shape[:2]
            return (bg_w - fg_w) // 2, (bg_h - fg_h) // 2

        return max_loc

    def _feature_matching(
        self, bg_frame: np.ndarray, fg_frame: np.ndarray
    ) -> tuple[int, int]:
        """Find alignment using feature matching.

        Args:
            bg_frame: Background frame
            fg_frame: Foreground frame

        Returns:
            (x, y) offset based on matched features
        """
        logger.debug("Using feature matching for spatial alignment")

        # Convert to grayscale
        bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
        fg_gray = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2GRAY)

        # Create ORB detector
        orb = cv2.ORB_create(nfeatures=1000)

        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(bg_gray, None)
        kp2, des2 = orb.detectAndCompute(fg_gray, None)

        if des1 is None or des2 is None:
            logger.warning("No features found, defaulting to center")
            bg_h, bg_w = bg_frame.shape[:2]
            fg_h, fg_w = fg_frame.shape[:2]
            return (bg_w - fg_w) // 2, (bg_h - fg_h) // 2

        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 10:
            logger.warning("Too few feature matches, defaulting to center")
            bg_h, bg_w = bg_frame.shape[:2]
            fg_h, fg_w = fg_frame.shape[:2]
            return (bg_w - fg_w) // 2, (bg_h - fg_h) // 2

        # Calculate average offset from good matches
        offsets = []
        for match in matches[:20]:  # Use top 20 matches
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            offset = (int(pt1[0] - pt2[0]), int(pt1[1] - pt2[1]))
            offsets.append(offset)

        # Use median offset for robustness
        x_offset = int(np.median([o[0] for o in offsets]))
        y_offset = int(np.median([o[1] for o in offsets]))

        logger.info(f"Feature matching found offset: ({x_offset}, {y_offset})")

        return x_offset, y_offset

    def get_frame_at_time(self, video_path: str, time_seconds: float) -> np.ndarray:
        """Extract a single frame from video at specified time.

        Args:
            video_path: Path to video file
            time_seconds: Time in seconds to extract frame

        Returns:
            Frame as numpy array
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        # Seek to time
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:  # Handle case where FPS might be 0
            logger.warning(
                f"Video {video_path} has FPS of 0. Cannot seek by time accurately."
            )
            frame_number = 0  # Default to first frame or handle error
        else:
            frame_number = int(fps * time_seconds)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read frame
        ret, frame = cap.read()
        cap.release()

        if not ret:
            logger.warning(
                f"Could not read frame at time {time_seconds}s from {video_path}"
            )
            raise IOError(f"Could not read frame at {time_seconds}s from {video_path}")

        return frame

    def extract_frames_sample(
        self, video_path: str, video_fps: float, target_sample_fps: float = 2.0
    ) -> List[np.ndarray]:
        """Extract sample frames from video for analysis.

        Args:
            video_path: Path to video file
            video_fps: Original FPS of the video
            target_sample_fps: Target number of frames to sample per second

        Returns:
            List of frames as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file for frame sampling: {video_path}")

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0

        # Calculate which frames to capture based on target_sample_fps
        if video_fps <= 0:  # Avoid division by zero
            logger.warning(
                f"Video {video_path} has invalid FPS ({video_fps}), cannot sample frames effectively."
            )
            cap.release()
            return []

        capture_interval = int(video_fps / target_sample_fps)
        if capture_interval == 0:  # if target_sample_fps is higher than video_fps
            capture_interval = 1

        # Calculate frames to extract
        frames_to_extract = []
        frame_idx = 0
        while frame_idx < total_frames:
            if frame_idx % capture_interval == 0:
                frames_to_extract.append(frame_idx)
            frame_idx += 1

        logger.info(
            f"Extracting {len(frames_to_extract)} sample frames from {Path(video_path).name} "
            f"(video: {total_frames} frames @ {video_fps:.2f} fps, duration: {duration:.2f}s)"
        )

        # Extract frames using seeking for better performance
        for i, frame_idx in enumerate(frames_to_extract):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                logger.debug(
                    f"Failed to read frame {frame_idx} from {Path(video_path).name}"
                )

            if i % 10 == 0:
                logger.debug(f"Extracted {i}/{len(frames_to_extract)} sample frames...")

        cap.release()
        logger.info(
            f"Successfully extracted {len(frames)} sample frames from {Path(video_path).name} "
            f"(sampled at approx {target_sample_fps} FPS)"
        )
        return frames

    def compute_frame_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute similarity between two frames using SSIM.

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            Similarity score (0-1)
        """
        # Resize frames to same size if needed
        if frame1.shape != frame2.shape:
            h = min(frame1.shape[0], frame2.shape[0])
            w = min(frame1.shape[1], frame2.shape[1])
            frame1 = cv2.resize(frame1, (w, h))
            frame2 = cv2.resize(frame2, (w, h))

        # Convert to grayscale for SSIM
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Compute SSIM
        score, _ = ssim(gray1, gray2, full=True)
        return score

    def find_keyframe_matches(
        self,
        bg_frames_sampled: List[np.ndarray],  # Renamed for clarity
        fg_frames_sampled: List[np.ndarray],  # Renamed for clarity
        bg_info: VideoInfo,  # Added for original FPS access
        fg_info: VideoInfo,  # Added for original FPS access
        target_sample_fps: float,  # Added to know sample interval
        threshold: float = 0.6,  # Adjusted threshold slightly
    ) -> List[Tuple[int, int, float]]:  # Returns original frame indices
        """Find matching keyframes between two videos.

        Args:
            bg_frames_sampled: Background video sampled frames
            fg_frames_sampled: Foreground video sampled frames
            bg_info: Background video metadata
            fg_info: Foreground video metadata
            target_sample_fps: The rate at which frames were sampled
            threshold: Minimum similarity threshold

        Returns:
            List of (original_bg_idx, original_fg_idx, similarity) tuples
        """
        logger.info("Finding keyframe matches between sampled video frames")
        matches = []

        bg_capture_interval = (
            int(bg_info.fps / target_sample_fps)
            if target_sample_fps > 0 and bg_info.fps > 0
            else 1
        )
        fg_capture_interval = (
            int(fg_info.fps / target_sample_fps)
            if target_sample_fps > 0 and fg_info.fps > 0
            else 1
        )
        if bg_capture_interval == 0:
            bg_capture_interval = 1
        if fg_capture_interval == 0:
            fg_capture_interval = 1

        # Search a wider window in background frames for each foreground frame
        for fg_sample_idx, fg_frame in enumerate(fg_frames_sampled):
            best_match_for_fg_sample = (-1, 0.0)  # (bg_sample_idx, similarity)

            # Search a large portion of bg_frames_sampled, e.g., +/- 25% of total bg_frames_sampled length
            # or even all of them if performance allows and initial sync is very off.
            # For now, let's try a large window or full scan for robustness.
            # Consider a window that adapts to the expected position.

            # Estimate expected position of fg_sample_idx in bg_frames_sampled
            # This rough estimate helps to center the search window if not scanning fully.
            proportional_idx = int(
                fg_sample_idx * len(bg_frames_sampled) / len(fg_frames_sampled)
            )

            # Define a search window. Example: +/- 10 sampled frames, or a percentage of total.
            # A large window like 30-50% of bg_frames_sampled around proportional_idx.
            # Or even full scan if not too many samples.
            window_radius_samples = (
                len(bg_frames_sampled) // 2
            )  # Search a very large window

            start_idx = max(0, proportional_idx - window_radius_samples)
            end_idx = min(
                len(bg_frames_sampled), proportional_idx + window_radius_samples + 1
            )

            # If the number of sampled frames is small, just search all of them.
            if len(bg_frames_sampled) < 50:  # Heuristic value
                start_idx = 0
                end_idx = len(bg_frames_sampled)

            for bg_sample_idx in range(start_idx, end_idx):
                similarity = self.compute_frame_similarity(
                    bg_frames_sampled[bg_sample_idx], fg_frame
                )
                if similarity > best_match_for_fg_sample[1]:
                    best_match_for_fg_sample = (bg_sample_idx, similarity)

            if best_match_for_fg_sample[1] >= threshold:
                original_bg_idx = best_match_for_fg_sample[0] * bg_capture_interval
                original_fg_idx = fg_sample_idx * fg_capture_interval
                matches.append(
                    (original_bg_idx, original_fg_idx, best_match_for_fg_sample[1])
                )
                logger.debug(
                    f"Keyframe match: BG_sample[{best_match_for_fg_sample[0]}] (orig:{original_bg_idx}) <-> FG_sample[{fg_sample_idx}] (orig:{original_fg_idx}) (similarity: {best_match_for_fg_sample[1]:.3f})"
                )

        # Sort matches by foreground frame index
        matches.sort(key=lambda x: x[1])

        # Filter out non-monotonic matches in background frames (optional, but can help)
        # If BG frame index for a later FG frame is earlier than for a previous FG frame, it's problematic.
        filtered_matches = []
        last_bg_idx = -1
        for bg_idx, fg_idx, sim in matches:
            if bg_idx >= last_bg_idx:
                filtered_matches.append((bg_idx, fg_idx, sim))
                last_bg_idx = bg_idx
            else:
                logger.debug(
                    f"Skipping non-monotonic match: BG[{bg_idx}] for FG[{fg_idx}] (last BG was {last_bg_idx})"
                )

        logger.info(
            f"Found {len(filtered_matches)} monotonic keyframe matches out of {len(matches)} total matches"
        )

        # Log match details for debugging
        if filtered_matches and self.verbose:
            logger.debug("Keyframe match details:")
            for i, (bg_idx, fg_idx, sim) in enumerate(
                filtered_matches[:5]
            ):  # Show first 5
                bg_time = bg_idx / bg_info.fps if bg_info.fps > 0 else 0
                fg_time = fg_idx / fg_info.fps if fg_info.fps > 0 else 0
                logger.debug(
                    f"  Match {i + 1}: BG[{bg_idx}]@{bg_time:.2f}s <-> FG[{fg_idx}]@{fg_time:.2f}s (sim={sim:.3f})"
                )
            if len(filtered_matches) > 5:
                logger.debug(f"  ... and {len(filtered_matches) - 5} more matches")

        return filtered_matches

    def compute_frame_alignment(
        self,
        bg_info: VideoInfo,
        fg_info: VideoInfo,
        keyframe_matches: List[Tuple[int, int, float]],  # Original frame indices
        trim: bool = False,  # Added trim argument
    ) -> Tuple[
        List[FrameAlignment], float
    ]:  # Returns list of alignments and an overall initial offset
        """Compute frame-by-frame alignment based on keyframe matches.
           This version creates a direct map from each FG frame to a BG frame.

        Args:
            bg_info: Background video info
            fg_info: Foreground video info
            keyframe_matches: List of (original_bg_idx, original_fg_idx, similarity) tuples
            trim: If true, only align frames within the span of keyframe_matches.

        Returns:
            (frame_alignments_list, initial_temporal_offset)
            frame_alignments_list: a list where each element is FrameAlignment(bg_frame_idx, fg_frame_idx, similarity_score, 0)
            initial_temporal_offset: The temporal offset of the first matched FG frame relative to BG start.
        """
        logger.info("Computing frame-by-frame alignment map from keyframe_matches")
        alignments = []

        if not keyframe_matches:
            logger.warning("No keyframe matches found. Cannot compute frame alignment.")
            return [], 0.0

        first_match_bg_time = keyframe_matches[0][0] / bg_info.fps
        first_match_fg_time = keyframe_matches[0][1] / fg_info.fps
        initial_temporal_offset = first_match_bg_time - first_match_fg_time
        logger.info(
            f"Initial temporal offset from first keyframe match: {initial_temporal_offset:.3f}s"
        )

        fg_start_frame_for_alignment = 0
        fg_end_frame_for_alignment = fg_info.frame_count

        if trim:
            fg_start_frame_for_alignment = keyframe_matches[0][
                1
            ]  # original_fg_idx of first match
            fg_end_frame_for_alignment = (
                keyframe_matches[-1][1] + 1
            )  # original_fg_idx of last match (exclusive end)
            logger.info(
                f"Trim active for frame alignment: processing FG frames from {fg_start_frame_for_alignment} to {fg_end_frame_for_alignment - 1}"
            )

        current_keyframe_idx = 0
        # Ensure current_keyframe_idx starts at the first keyframe relevant to fg_start_frame_for_alignment
        while (
            current_keyframe_idx < len(keyframe_matches)
            and keyframe_matches[current_keyframe_idx][1] < fg_start_frame_for_alignment
        ):
            current_keyframe_idx += 1
        if (
            current_keyframe_idx > 0
            and keyframe_matches[current_keyframe_idx - 1][1]
            < fg_start_frame_for_alignment
        ):
            # This means fg_start_frame_for_alignment is between keyframe_matches[current_keyframe_idx-1] and keyframe_matches[current_keyframe_idx]
            pass  # current_keyframe_idx is correctly pointing to the *next* keyframe or is at 0

        for fg_frame_num in range(
            fg_start_frame_for_alignment, fg_end_frame_for_alignment
        ):
            kf_prev = None
            kf_next = None

            # Advance current_keyframe_idx to find the segment for fg_frame_num
            # We need keyframe_matches[idx-1].fg_idx <= fg_frame_num <= keyframe_matches[idx].fg_idx
            temp_search_idx = current_keyframe_idx
            while (
                temp_search_idx < len(keyframe_matches)
                and keyframe_matches[temp_search_idx][1] < fg_frame_num
            ):
                temp_search_idx += 1

            # After loop, keyframe_matches[temp_search_idx] is the first one with .fg_idx >= fg_frame_num
            # Or temp_search_idx is len(keyframe_matches)

            if temp_search_idx == 0:  # fg_frame_num is before or at the first keyframe
                kf_prev = keyframe_matches[0]
                kf_next = keyframe_matches[0]
            elif temp_search_idx == len(
                keyframe_matches
            ):  # fg_frame_num is after or at the last keyframe
                kf_prev = keyframe_matches[-1]
                kf_next = keyframe_matches[-1]
            else:  # fg_frame_num is between two keyframes or at keyframe_matches[temp_search_idx]
                kf_prev = keyframe_matches[temp_search_idx - 1]
                kf_next = keyframe_matches[temp_search_idx]

            # Update current_keyframe_idx for next iteration's starting point (small optimization)
            current_keyframe_idx = max(
                0, temp_search_idx - 1
            )  # Start search for next fg_frame_num near here

            fg_prev_idx, bg_prev_idx, sim_prev = kf_prev[1], kf_prev[0], kf_prev[2]
            fg_next_idx, bg_next_idx, sim_next = kf_next[1], kf_next[0], kf_next[2]

            mapped_bg_frame_num = 0
            current_similarity = 0.5

            if fg_next_idx == fg_prev_idx:
                # At a keyframe or extrapolating from a single keyframe (start/end of sequence)
                # Calculate bg_frame_num by maintaining relative distance from the keyframe
                # Use the average frame rate ratio for better extrapolation
                fps_ratio = bg_info.fps / fg_info.fps if fg_info.fps > 0 else 1.0
                mapped_bg_frame_num = int(
                    bg_prev_idx + (fg_frame_num - fg_prev_idx) * fps_ratio
                )
                current_similarity = sim_prev
            else:
                # Interpolate between keyframes
                # Use a smoothed interpolation that considers the frame rate differences
                ratio = (fg_frame_num - fg_prev_idx) / (fg_next_idx - fg_prev_idx)

                # Apply a smoothing function to the ratio to reduce abrupt changes
                # Using a cubic smoothing function for more natural transitions
                smooth_ratio = (
                    ratio * ratio * (3.0 - 2.0 * ratio)
                )  # Smoothstep function

                mapped_bg_frame_num = int(
                    bg_prev_idx + smooth_ratio * (bg_next_idx - bg_prev_idx)
                )
                current_similarity = sim_prev + smooth_ratio * (sim_next - sim_prev)

            mapped_bg_frame_num = max(
                0, min(mapped_bg_frame_num, bg_info.frame_count - 1)
            )

            alignments.append(
                FrameAlignment(
                    bg_frame_idx=mapped_bg_frame_num,
                    fg_frame_idx=fg_frame_num,
                    similarity_score=current_similarity,
                    temporal_offset=0,
                )
            )

        logger.info(
            f"Generated frame-to-frame alignment map for {len(alignments)} FG frames."
        )

        # Log some statistics about the alignment
        if alignments and self.verbose:
            # Calculate average frame step ratio
            bg_steps = []
            fg_steps = []
            for i in range(1, min(100, len(alignments))):  # Sample first 100 frames
                bg_steps.append(
                    alignments[i].bg_frame_idx - alignments[i - 1].bg_frame_idx
                )
                fg_steps.append(
                    alignments[i].fg_frame_idx - alignments[i - 1].fg_frame_idx
                )

            if bg_steps and fg_steps:
                avg_bg_step = sum(bg_steps) / len(bg_steps)
                avg_fg_step = sum(fg_steps) / len(fg_steps)
                step_ratio = avg_bg_step / avg_fg_step if avg_fg_step > 0 else 0
                logger.debug(f"Average frame step ratio (BG/FG): {step_ratio:.3f}")
                logger.debug(
                    f"Expected FPS ratio (BG/FG): {bg_info.fps / fg_info.fps if fg_info.fps > 0 else 0:.3f}"
                )

        return alignments, initial_temporal_offset

    def compose_videos(
        self,
        bg_path: str,
        fg_path: str,
        output_path: str,
        spatial_offset: tuple[int, int],
        temporal_offset: float,
        bg_info: VideoInfo,
        fg_info: VideoInfo,
        use_bg_audio: bool = True,
        frame_alignments: List[FrameAlignment] | None = None,
        trim: bool = False,
    ):
        """Compose the final video using FFmpeg.

        Args:
            bg_path: Background video path
            fg_path: Foreground video path
            output_path: Output video path
            spatial_offset: (x, y) position for overlay
            temporal_offset: Time offset in seconds
            bg_info: Background video metadata
            fg_info: Foreground video metadata
            use_bg_audio: Whether to use background audio
            frame_alignments: Optional frame-by-frame alignment data
            trim: Whether to trim output to overlapping segments only
        """
        logger.info("Composing final video")

        # Determine output frame rate (use higher)
        output_fps = max(bg_info.fps, fg_info.fps)

        # Handle trimming if requested
        trim_start = 0.0
        trim_duration = None

        if trim and temporal_offset > 0:
            # Trim beginning of background to start where foreground starts
            trim_start = temporal_offset
            trim_duration = min(bg_info.duration - temporal_offset, fg_info.duration)
            logger.info(
                f"Trimming output: start={trim_start:.3f}s, duration={trim_duration:.3f}s"
            )

        # Build FFmpeg command
        if trim and trim_start > 0:
            inputs = [ffmpeg.input(bg_path, ss=trim_start)]
        else:
            inputs = [ffmpeg.input(bg_path)]

        # Add foreground with time offset if needed
        if temporal_offset > 0 and not trim:
            fg_input = ffmpeg.input(fg_path, itsoffset=temporal_offset)
        else:
            fg_input = ffmpeg.input(fg_path)
        inputs.append(fg_input)

        # Build filter graph
        bg_video = inputs[0]["v"]
        fg_video = inputs[1]["v"]

        # Scale foreground if needed
        if fg_info.width > bg_info.width or fg_info.height > bg_info.height:
            # Scale to fit within background
            scale_factor = min(
                bg_info.width / fg_info.width, bg_info.height / fg_info.height
            )
            new_width = int(fg_info.width * scale_factor)
            new_height = int(fg_info.height * scale_factor)

            fg_video = fg_video.filter("scale", new_width, new_height)
            logger.info(f"Scaling foreground to {new_width}x{new_height}")

        # Apply overlay
        x, y = spatial_offset
        video = ffmpeg.filter(
            [bg_video, fg_video],
            "overlay",
            x=x,
            y=y,
            eof_action="pass",  # Continue background after foreground ends
        )

        # Handle audio
        if bg_info.has_audio and use_bg_audio:
            audio = inputs[0]["a"]
        elif fg_info.has_audio and not use_bg_audio:
            audio = inputs[1]["a"]
        else:
            audio = None

        # Build output
        output_args = {
            "c:v": "libx264",
            "preset": "medium",
            "crf": 18,
            "r": output_fps,
            "pix_fmt": "yuv420p",
        }

        if trim and trim_duration:
            output_args["t"] = trim_duration

        if audio is not None:
            output_args["c:a"] = "aac"
            output_args["b:a"] = "192k"
            output = ffmpeg.output(video, audio, output_path, **output_args)
        else:
            output = ffmpeg.output(video, output_path, **output_args)

        # Run FFmpeg
        try:
            ffmpeg.run(output, overwrite_output=True, capture_stderr=True)
            logger.info(f"Video saved to: {output_path}")
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise

    def _merge_audio_with_ffmpeg(
        self,
        video_path_no_audio: str,
        audio_source_path: str,
        output_path: str,
        audio_offset: float = 0.0,  # Offset for the audio stream
        video_duration: float | None = None,
        target_fps: float | None = None,
    ):
        """Merge audio from audio_source_path into video_path_no_audio using FFmpeg."""
        logger.info(
            f"Merging audio from {Path(audio_source_path).name} into video, saving to {output_path}"
        )

        input_video = ffmpeg.input(video_path_no_audio)
        # input_audio = ffmpeg.input(audio_source_path)
        # Redefine input_audio with itsoffset for clarity and correctness
        effective_audio_input_args = {}
        if audio_offset != 0:
            logger.debug(
                f"Applying audio offset of {audio_offset:.3f}s to {Path(audio_source_path).name}"
            )
            effective_audio_input_args["itsoffset"] = str(audio_offset)

        input_audio = ffmpeg.input(audio_source_path, **effective_audio_input_args)

        video_stream = input_video["v"]
        audio_stream = input_audio["a"]

        output_args = {
            "c:v": "copy",  # Copy video stream as is
            "c:a": "aac",
            "b:a": "192k",
            "shortest": None,
        }
        if video_duration:
            output_args["t"] = str(video_duration)  # Ensure it's a string for ffmpeg
        # target_fps is for video stream, but we copy video, so not directly used here.

        try:
            output_path_str = str(output_path)
            stream = ffmpeg.output(
                video_stream, audio_stream, output_path_str, **output_args
            )
            ffmpeg.run(stream, overwrite_output=True, capture_stderr=True)
            logger.info(f"Video with merged audio saved to: {output_path_str}")
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg audio merge error: {e.stderr.decode()}")
            try:
                Path(video_path_no_audio).rename(output_path_str)
                logger.warning(
                    f"Audio merge failed. Saved video without audio to {output_path_str}"
                )
            except Exception as ren_err:
                logger.error(
                    f"Failed to rename video after audio merge failure: {ren_err}"
                )
            raise

    def compose_videos_opencv(
        self,
        bg_info: VideoInfo,
        fg_info: VideoInfo,
        output_path_str: str,  # Final output path
        spatial_offset: tuple[int, int],
        frame_alignments: List[FrameAlignment],
        trim: bool = False,
    ):
        """Compose final video using OpenCV for frame-by-frame overlay based on alignments.
           The resulting video will be silent. Audio needs to be merged separately.
        Args:
            bg_info: Background video metadata.
            fg_info: Foreground video metadata.
            output_path_str: Path to save the silent composited video.
            spatial_offset: (x,y) for foreground placement.
            frame_alignments: List of FrameAlignment objects mapping fg to bg frames.
            trim: Trim output to overlapping segments.
        Returns:
            True if composition successful, False otherwise.
        """
        logger.info(f"Composing video with OpenCV, outputting to {output_path_str}")

        if not frame_alignments:
            logger.error(
                "No frame alignments provided for OpenCV composition. Aborting."
            )
            return False

        output_fps = fg_info.fps  # Output FPS driven by foreground video's rate
        output_width = bg_info.width
        output_height = bg_info.height

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            output_path_str, fourcc, output_fps, (output_width, output_height)
        )

        if not video_writer.isOpened():
            logger.error(f"Failed to open VideoWriter for {output_path_str}")
            return False

        cap_bg = cv2.VideoCapture(bg_info.path)
        cap_fg = cv2.VideoCapture(fg_info.path)

        if not cap_bg.isOpened() or not cap_fg.isOpened():
            logger.error(
                "Failed to open background or foreground video for OpenCV composition."
            )
            if cap_bg.isOpened():
                cap_bg.release()
            if cap_fg.isOpened():
                cap_fg.release()
            video_writer.release()
            return False

        actual_frames_written = 0

        # Determine start and end FG frames for trimming
        if trim and frame_alignments:
            # If trim, only process the aligned segment based on keyframe matches
            # This means the frame_alignments list itself should represent the trimmed segment.
            # The calling code should prepare frame_alignments to only contain trimmed parts if trim is true.
            # Here, we assume frame_alignments IS the segment to render.
            pass

        # Use existing console for progress updates instead of creating new Progress context
        logger.info(f"Starting OpenCV composition of {len(frame_alignments)} frames")

        # Track progress without nested Progress context
        progress_interval = max(1, len(frame_alignments) // 20)  # Update every 5%

        for align_info in frame_alignments:
            fg_frame_to_get = align_info.fg_frame_idx
            bg_frame_to_get = align_info.bg_frame_idx

            cap_fg.set(cv2.CAP_PROP_POS_FRAMES, fg_frame_to_get)
            ret_fg, fg_frame_img = cap_fg.read()

            cap_bg.set(cv2.CAP_PROP_POS_FRAMES, bg_frame_to_get)
            ret_bg, bg_frame_img = cap_bg.read()

            if not ret_fg or not ret_bg:
                logger.warning(
                    f"Failed to read FG frame {fg_frame_to_get} (ret={ret_fg}) or BG frame {bg_frame_to_get} (ret={ret_bg}). Skipping."
                )
                continue

            current_fg_h, current_fg_w = fg_frame_img.shape[:2]
            final_fg_frame = fg_frame_img

            if current_fg_w > bg_info.width or current_fg_h > bg_info.height:
                scale_factor = min(
                    bg_info.width / current_fg_w, bg_info.height / current_fg_h
                )
                new_w = int(current_fg_w * scale_factor)
                new_h = int(current_fg_h * scale_factor)
                final_fg_frame = cv2.resize(
                    fg_frame_img, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
                current_fg_h, current_fg_w = new_h, new_w

            x_offset, y_offset = spatial_offset
            composited_frame = bg_frame_img.copy()

            roi_x_start = max(0, x_offset)
            roi_y_start = max(0, y_offset)
            roi_x_end = min(bg_info.width, x_offset + current_fg_w)
            roi_y_end = min(bg_info.height, y_offset + current_fg_h)

            fg_crop_x_start = 0
            fg_crop_y_start = 0
            if x_offset < 0:
                fg_crop_x_start = abs(x_offset)
            if y_offset < 0:
                fg_crop_y_start = abs(y_offset)

            fg_to_overlay_w = roi_x_end - roi_x_start
            fg_to_overlay_h = roi_y_end - roi_y_start

            if fg_to_overlay_w > 0 and fg_to_overlay_h > 0:
                fg_cropped_for_overlay = final_fg_frame[
                    fg_crop_y_start : fg_crop_y_start + fg_to_overlay_h,
                    fg_crop_x_start : fg_crop_x_start + fg_to_overlay_w,
                ]

                if (
                    fg_cropped_for_overlay.shape[0] == fg_to_overlay_h
                    and fg_cropped_for_overlay.shape[1] == fg_to_overlay_w
                ):
                    composited_frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = (
                        fg_cropped_for_overlay
                    )
                else:
                    logger.warning(
                        f"Skipping overlay for FG frame {fg_frame_to_get} due to unexpected crop size. "
                        f"ROI: {fg_to_overlay_w}x{fg_to_overlay_h}, Cropped: {fg_cropped_for_overlay.shape[1]}x{fg_cropped_for_overlay.shape[0]}"
                    )
            else:
                logger.debug(
                    f"FG frame {fg_frame_to_get} is completely outside BG frame after offset. Writing BG frame only."
                )

            video_writer.write(composited_frame)
            actual_frames_written += 1

            # Log progress at intervals
            if actual_frames_written % progress_interval == 0:
                progress_pct = (actual_frames_written / len(frame_alignments)) * 100
                logger.info(
                    f"OpenCV composition progress: {progress_pct:.1f}% ({actual_frames_written}/{len(frame_alignments)} frames)"
                )

        logger.info(
            f"OpenCV composition finished. {actual_frames_written} frames written to {output_path_str}."
        )
        cap_bg.release()
        cap_fg.release()
        video_writer.release()
        return True

    def process(
        self,
        bg: str,
        fg: str,
        output: str | None = None,
        match_space: str = "precise",
        temporal_align: str = "audio",
        trim: bool = False,
        skip_spatial_align: bool = False,
        max_keyframes: int = 2000,
    ):
        """Main processing function.

        Args:
            bg: Background video path
            fg: Foreground video path
            output: Output video path (auto-generated if not provided)
            match_space: Spatial alignment method ('precise' or 'fast')
            temporal_align: Temporal alignment method ('audio', 'duration', or 'frames')
            trim: Trim output to overlapping segments only
            skip_spatial_align: Skip spatial alignment (use center)
            max_keyframes: Maximum number of keyframes to use in frames mode
        """
        # Validate inputs
        bg_path = Path(bg)
        fg_path = Path(fg)

        if not bg_path.exists():
            logger.error(f"Background video not found: {bg}")
            return

        if not fg_path.exists():
            logger.error(f"Foreground video not found: {fg}")
            return

        # Generate output path if not provided
        if output is None:
            output = bg_path.stem + "_overlay_" + fg_path.stem + ".mp4"
            logger.info(f"Output path: {output}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Get video info
            task = progress.add_task("Analyzing videos...", total=None)
            logger.info(
                f"Processing overlay: {Path(bg_path).name} + {Path(fg_path).name}"
            )
            bg_info = self.get_video_info(str(bg_path))
            fg_info = self.get_video_info(str(fg_path))
            progress.update(task, completed=True)

            # Log video compatibility
            logger.info("Video compatibility check:")
            logger.info(
                f"  Resolution: BG {bg_info.width}x{bg_info.height} vs FG {fg_info.width}x{fg_info.height}"
            )
            logger.info(f"  Frame rate: BG {bg_info.fps:.2f} vs FG {fg_info.fps:.2f}")
            logger.info(
                f"  Duration: BG {bg_info.duration:.2f}s vs FG {fg_info.duration:.2f}s"
            )
            logger.info(
                f"  Audio: BG {'yes' if bg_info.has_audio else 'no'} vs FG {'yes' if fg_info.has_audio else 'no'}"
            )

            # Temporal alignment
            temporal_offset = 0.0
            frame_alignments = None
            alignment_mode = AlignmentMode(temporal_align)

            if (
                alignment_mode == AlignmentMode.AUDIO
                and bg_info.has_audio
                and fg_info.has_audio
            ):
                task = progress.add_task("Aligning by audio...", total=None)

                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as bg_audio:
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    ) as fg_audio:
                        try:
                            # Extract audio
                            if self.extract_audio(str(bg_path), bg_audio.name):
                                if self.extract_audio(str(fg_path), fg_audio.name):
                                    temporal_offset = self.compute_audio_offset(
                                        bg_audio.name, fg_audio.name
                                    )

                                    # Clamp negative offsets to 0
                                    if temporal_offset < 0:
                                        logger.warning(
                                            f"Negative offset {temporal_offset:.3f}s, "
                                            "clamping to 0"
                                        )
                                        temporal_offset = 0.0
                        finally:
                            # Clean up temp files
                            Path(bg_audio.name).unlink(missing_ok=True)
                            Path(fg_audio.name).unlink(missing_ok=True)

                progress.update(task, completed=True)

            elif alignment_mode == AlignmentMode.FRAMES:
                task = progress.add_task("Aligning by frames...", total=None)

                # Calculate target sample FPS based on max_keyframes
                # We want to sample enough frames but not exceed max_keyframes
                min_frame_count = min(bg_info.frame_count, fg_info.frame_count)
                min_duration = min(bg_info.duration, fg_info.duration)

                # Calculate the ideal sample FPS to get around max_keyframes/2 samples
                # (we use /2 because we want room for both videos)
                if min_duration > 0:
                    target_sample_fps = min(
                        max_keyframes / 2.0 / min_duration,
                        min_frame_count / min_duration,
                    )
                    # Ensure at least 1 fps sampling, but cap at reasonable rate
                    target_sample_fps = max(1.0, min(30.0, target_sample_fps))
                else:
                    target_sample_fps = 2.0  # Fallback

                logger.info(
                    f"Frame-based alignment selected. Extracting sample frames (target {target_sample_fps:.1f} FPS)"
                )
                logger.info(f"Max keyframes allowed: {max_keyframes}")
                logger.debug(
                    f"Background video: {bg_info.frame_count} frames @ {bg_info.fps:.2f} fps"
                )
                logger.debug(
                    f"Foreground video: {fg_info.frame_count} frames @ {fg_info.fps:.2f} fps"
                )

                bg_frames_sampled = self.extract_frames_sample(
                    str(bg_path), bg_info.fps, target_sample_fps
                )
                fg_frames_sampled = self.extract_frames_sample(
                    str(fg_path), fg_info.fps, target_sample_fps
                )

                if not bg_frames_sampled or not fg_frames_sampled:
                    logger.error(
                        "Could not extract sample frames. Falling back to duration alignment."
                    )
                    alignment_mode = AlignmentMode.DURATION  # Fallback
                else:
                    keyframe_matches = self.find_keyframe_matches(
                        bg_frames_sampled,
                        fg_frames_sampled,
                        bg_info,
                        fg_info,
                        target_sample_fps,
                    )

                    if not keyframe_matches:
                        logger.warning(
                            "No keyframe matches found from sampled frames. Falling back to duration alignment."
                        )
                        alignment_mode = AlignmentMode.DURATION  # Fallback
                    else:
                        frame_alignments_list, frame_based_initial_offset = (
                            self.compute_frame_alignment(
                                bg_info, fg_info, keyframe_matches, trim
                            )
                        )
                        if not frame_alignments_list:
                            logger.warning(
                                "Frame alignment list is empty (possibly due to trim or no matches). Falling back to duration alignment."
                            )
                            alignment_mode = AlignmentMode.DURATION  # Fallback
                        else:
                            frame_alignments = frame_alignments_list
                            temporal_offset = frame_based_initial_offset
                            logger.info(
                                f"Using frame-based initial temporal offset: {temporal_offset:.3f}s for subsequent steps."
                            )
                progress.update(task, completed=True)

            elif alignment_mode == AlignmentMode.DURATION:
                # Simple duration-based alignment (center fg within bg duration)
                logger.info(
                    "Using duration-based alignment (or fallback for frames mode)"
                )
                if bg_info.duration > fg_info.duration:
                    temporal_offset = (bg_info.duration - fg_info.duration) / 2
                else:
                    temporal_offset = 0.0
                logger.info(f"Duration-based offset: {temporal_offset:.3f}s")

            else:
                logger.info(f"No temporal alignment applied (mode: {temporal_align})")

            # Spatial alignment (common for all modes, uses the determined temporal_offset)
            spatial_offset = (0, 0)
            if not skip_spatial_align:
                task = progress.add_task("Aligning spatially...", total=None)
                try:
                    # Get frames for comparison using the determined temporal_offset
                    fg_frame_time_in_fg = 1.0  # e.g., 1 second into FG video
                    bg_frame_time_in_bg = fg_frame_time_in_fg + temporal_offset

                    # Ensure bg_frame_time_in_bg is valid
                    if (
                        bg_frame_time_in_bg < 0
                        or bg_frame_time_in_bg >= bg_info.duration
                    ):
                        logger.warning(
                            f"Calculated BG frame time {bg_frame_time_in_bg:.2f}s is out of bounds for spatial alignment. Using BG start."
                        )
                        bg_frame_time_in_bg = (
                            1.0  # Fallback to a frame near start of BG
                        )

                    fg_frame = self.get_frame_at_time(str(fg_path), fg_frame_time_in_fg)
                    bg_frame = self.get_frame_at_time(str(bg_path), bg_frame_time_in_bg)

                    spatial_offset = self.compute_spatial_offset(
                        bg_frame, fg_frame, method=match_space
                    )
                except IOError as e:
                    logger.warning(
                        f"Could not read frames for spatial alignment: {e}. Defaulting to center."
                    )
                    spatial_offset = (
                        (bg_info.width - fg_info.width) // 2,
                        (bg_info.height - fg_info.height) // 2,
                    )
                progress.update(task, completed=True)
            else:
                spatial_offset = (
                    (bg_info.width - fg_info.width) // 2,
                    (bg_info.height - fg_info.height) // 2,
                )
                logger.info(
                    f"Spatial alignment skipped. Using center position: {spatial_offset}"
                )

            # --- Video Composition Stage ---
            final_output_path = Path(output)
            task = progress.add_task(
                "Composing video...", total=None
            )  # Main composition task

            audio_source_for_merge = None
            use_bg_audio = True  # Default
            if bg_info.has_audio and fg_info.has_audio:
                use_bg_audio = True
                audio_source_for_merge = bg_info.path
            elif not bg_info.has_audio and fg_info.has_audio:
                use_bg_audio = False
                audio_source_for_merge = fg_info.path
            elif bg_info.has_audio and not fg_info.has_audio:
                use_bg_audio = True
                audio_source_for_merge = bg_info.path

            # `temporal_offset` here is the one determined by audio, duration, or the *initial* one from frames mode.

            if alignment_mode == AlignmentMode.FRAMES and frame_alignments:
                logger.info(
                    f"Using OpenCV for frame-by-frame composition ({len(frame_alignments)} frame mappings)"
                )
                temp_video_path = final_output_path.with_suffix(
                    ".temp_opencv_video.mp4"
                )

                compose_success = self.compose_videos_opencv(
                    bg_info,
                    fg_info,
                    str(temp_video_path),
                    spatial_offset,
                    frame_alignments,
                    trim,
                )
                progress.update(
                    task, completed=True
                )  # Mark main composing task as done for video part

                if compose_success and audio_source_for_merge:
                    audio_merge_task = progress.add_task(
                        "Merging audio (FFmpeg)...", total=None
                    )
                    # The `temporal_offset` is the initial overall offset of FG relative to BG.
                    # This `temporal_offset` should be applied to the `audio_source_for_merge` if it's the BG audio.
                    # If FG audio is used, its offset should be 0 relative to its own start, because frame_alignments map FG frames.
                    first_aligned_bg_frame_idx = frame_alignments[0].bg_frame_idx
                    audio_offset_for_final_merge = (
                        first_aligned_bg_frame_idx / bg_info.fps
                    )
                    logger.info(
                        f"Frame mode using BG audio: Audio source {Path(audio_source_for_merge).name} will be started from {audio_offset_for_final_merge:.3f}s for merge."
                    )

                    self._merge_audio_with_ffmpeg(
                        str(temp_video_path),
                        audio_source_for_merge,
                        str(final_output_path),
                        audio_offset=audio_offset_for_final_merge,
                        video_duration=len(frame_alignments) / fg_info.fps,
                        target_fps=fg_info.fps,
                    )
                    if temp_video_path.exists():
                        temp_video_path.unlink(missing_ok=True)
                    progress.update(audio_merge_task, completed=True)
                elif compose_success:
                    logger.info(
                        "OpenCV composition successful. No audio source to merge or audio merge skipped. Renaming temp video."
                    )
                    if temp_video_path.exists():
                        temp_video_path.rename(final_output_path)
                else:
                    logger.error(
                        "OpenCV video composition failed. No output generated."
                    )
                    if temp_video_path.exists():
                        temp_video_path.unlink(
                            missing_ok=True
                        )  # Clean up failed attempt

            else:
                logger.info(
                    "Using FFmpeg for composition (audio/duration mode or frames fallback)."
                )
                # `temporal_offset` here is from audio, duration, or initial frame based offset if frames mode fell back early
                self.compose_videos(
                    str(bg_path),
                    str(fg_path),
                    str(final_output_path),  # Ensure output is string
                    spatial_offset,
                    temporal_offset,  # This is the global offset for FFmpeg
                    bg_info,
                    fg_info,
                    use_bg_audio,
                    None,  # FFmpeg path does not use detailed frame_alignments map
                    trim,
                )
                progress.update(task, completed=True)

        # Final summary
        logger.info("Overlay processing complete:")
        logger.info(f"  Method: {alignment_mode.value} alignment")
        logger.info(f"  Spatial offset: {spatial_offset}")
        logger.info(f"  Temporal offset: {temporal_offset:.3f}s")
        if frame_alignments:
            logger.info(f"  Frame mappings: {len(frame_alignments)}")
        logger.info(f"  Trim: {'enabled' if trim else 'disabled'}")
        logger.info(f"  Output: {final_output_path}")

        console.print(f"[green]Video overlay complete: {final_output_path}")


def main(
    bg: str,
    fg: str,
    output: str | None = None,
    match_space: str = "precise",
    temporal_align: str = "frames",
    trim: bool = True,
    skip_spatial_align: bool = False,
    verbose: bool = False,
    max_keyframes: int = 2000,
):
    """Overlay foreground video onto background video with intelligent alignment.

    Args:
        bg: Background video path
        fg: Foreground video path
        output: Output video path (auto-generated if not provided)
        match_space: Spatial alignment method ('precise' or 'fast')
        temporal_align: Temporal alignment method ('audio', 'duration', or 'frames') [default: 'frames']
        trim: Trim output to overlapping segments only [default: True]
        skip_spatial_align: Skip spatial alignment (use center)
        verbose: Enable verbose logging
        max_keyframes: Maximum number of keyframes to use in frames mode [default: 2000]
    """
    processor = VideoOverlay(verbose=verbose)
    processor.process(
        bg,
        fg,
        output,
        match_space,
        temporal_align,
        trim,
        skip_spatial_align,
        max_keyframes,
    )


if __name__ == "__main__":
    fire.Fire(main)
