#!/usr/bin/env python3
# this_file: src/vidkompy/core/video_processor.py

"""
Core video processing functionality.

Handles video I/O, metadata extraction, and frame operations.
"""

import cv2
import ffmpeg
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console

from ..models import VideoInfo

console = Console()


class VideoProcessor:
    """Handles core video processing operations."""
    
    def get_video_info(self, video_path: str) -> VideoInfo:
        """Extract video metadata using ffprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoInfo object with metadata
            
        Raises:
            ValueError: If video cannot be probed
        """
        logger.debug(f"Probing video: {video_path}")
        
        try:
            probe = ffmpeg.probe(video_path)
            
            # Find video stream
            video_stream = next(
                (s for s in probe["streams"] if s["codec_type"] == "video"), 
                None
            )
            
            if not video_stream:
                raise ValueError(f"No video stream found in {video_path}")
            
            # Extract properties
            width = int(video_stream["width"])
            height = int(video_stream["height"])
            
            # Parse frame rate
            fps_str = video_stream.get("r_frame_rate", "0/1")
            if "/" in fps_str:
                num, den = map(int, fps_str.split("/"))
                fps = num / den if den != 0 else 0
            else:
                fps = float(fps_str)
            
            duration = float(probe["format"].get("duration", 0))
            
            # Calculate frame count
            frame_count = int(video_stream.get("nb_frames", 0))
            if frame_count == 0 and duration > 0 and fps > 0:
                frame_count = int(duration * fps)
            
            # Check audio
            audio_stream = next(
                (s for s in probe["streams"] if s["codec_type"] == "audio"),
                None
            )
            
            has_audio = audio_stream is not None
            audio_sample_rate = None
            audio_channels = None
            
            if audio_stream:
                audio_sample_rate = int(audio_stream.get("sample_rate", 0))
                audio_channels = int(audio_stream.get("channels", 0))
            
            info = VideoInfo(
                width=width,
                height=height,
                fps=fps,
                duration=duration,
                frame_count=frame_count,
                has_audio=has_audio,
                audio_sample_rate=audio_sample_rate,
                audio_channels=audio_channels,
                path=video_path
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
    
    def extract_frames(
        self, 
        video_path: str, 
        frame_indices: List[int],
        resize_factor: float = 1.0
    ) -> List[np.ndarray]:
        """Extract specific frames from video.
        
        Args:
            video_path: Path to video file
            frame_indices: List of frame indices to extract
            resize_factor: Factor to resize frames (for performance)
            
        Returns:
            List of frames as numpy arrays
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return frames
        
        try:
            # Only show progress for large frame extractions
            if len(frame_indices) > 50:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    console=console,
                    transient=True
                ) as progress:
                    task = progress.add_task(
                        f"Extracting {len(frame_indices)} frames...",
                        total=len(frame_indices)
                    )
                    
                    for idx in frame_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        
                        if ret:
                            if resize_factor != 1.0:
                                height, width = frame.shape[:2]
                                new_width = int(width * resize_factor)
                                new_height = int(height * resize_factor)
                                frame = cv2.resize(frame, (new_width, new_height))
                            frames.append(frame)
                        else:
                            logger.warning(f"Failed to read frame {idx} from {video_path}")
                        
                        progress.update(task, advance=1)
            else:
                # No progress bar for small extractions
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        if resize_factor != 1.0:
                            height, width = frame.shape[:2]
                            new_width = int(width * resize_factor)
                            new_height = int(height * resize_factor)
                            frame = cv2.resize(frame, (new_width, new_height))
                        frames.append(frame)
                    else:
                        logger.warning(f"Failed to read frame {idx} from {video_path}")
                    
        finally:
            cap.release()
            
        return frames
    
    def extract_frame_range(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        step: int = 1,
        resize_factor: float = 1.0
    ) -> List[Tuple[int, np.ndarray]]:
        """Extract a range of frames with their indices.
        
        Args:
            video_path: Path to video
            start_frame: Starting frame index
            end_frame: Ending frame index (exclusive)
            step: Frame step size
            resize_factor: Resize factor for frames
            
        Returns:
            List of (frame_index, frame) tuples
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return frames
        
        try:
            for idx in range(start_frame, end_frame, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    if resize_factor != 1.0:
                        height, width = frame.shape[:2]
                        new_width = int(width * resize_factor)
                        new_height = int(height * resize_factor)
                        frame = cv2.resize(frame, (new_width, new_height))
                    frames.append((idx, frame))
                else:
                    break
                    
        finally:
            cap.release()
            
        return frames
    
    def extract_audio(
        self, 
        video_path: str, 
        output_path: str,
        sample_rate: int = 16000
    ) -> bool:
        """Extract audio from video to WAV file.
        
        Args:
            video_path: Input video path
            output_path: Output WAV path
            sample_rate: Target sample rate
            
        Returns:
            True if extraction successful
        """
        logger.debug(f"Extracting audio from {video_path}")
        
        try:
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(
                stream,
                output_path,
                acodec="pcm_s16le",
                ac=1,  # Mono
                ar=sample_rate,
                loglevel="error"
            )
            ffmpeg.run(stream, overwrite_output=True)
            return True
            
        except ffmpeg.Error as e:
            logger.error(f"Audio extraction failed: {e.stderr.decode()}")
            return False
    
    def create_video_writer(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "mp4v"
    ) -> cv2.VideoWriter:
        """Create OpenCV video writer.
        
        Args:
            output_path: Output video path
            width: Video width
            height: Video height  
            fps: Frame rate
            codec: Video codec (default mp4v)
            
        Returns:
            VideoWriter object
        """
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            raise ValueError(f"Failed to create video writer for {output_path}")
            
        return writer