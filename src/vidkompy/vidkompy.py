#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["fire", "rich", "loguru", "opencv-python", "numpy", "scipy", "ffmpeg-python", "soundfile", "scikit-image"]
# ///
# this_file: src/vidkompy/vidkompy.py

"""
Intelligent video overlay tool with automatic spatial and temporal alignment.

This tool overlays foreground videos onto background videos with smart alignment:
- Preserves all foreground frames without retiming
- Finds optimal background frames for each foreground frame
- Supports audio-based and frame-based temporal alignment
- Automatic spatial alignment with template/feature matching
"""

import sys
import fire
from loguru import logger
from pathlib import Path

from .core.video_processor import VideoProcessor
from .core.alignment_engine import AlignmentEngine
from .models import MatchTimeMode, TemporalMethod


def main(
    bg: str,
    fg: str,
    output: str | None = None,
    border: int = 8,
    blend: bool = False,
    gpu: bool = False,  # Future GPU acceleration support
    verbose: bool = False,
):
    """Overlay foreground video onto background video with intelligent alignment.

    Args:
        bg: Background video path
        fg: Foreground video path
        output: Output video path (auto-generated if not provided)
        border: Border thickness for border matching mode (default: 8)
        blend: Enable smooth blending at frame edges
        gpu: Enable GPU acceleration (future feature)
        verbose: Enable verbose logging
    """
    # Setup logging
    logger.remove()
    if verbose:
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

    # Validate inputs
    bg_path = Path(bg)
    fg_path = Path(fg)

    if not bg_path.exists():
        logger.error(f"Background video not found: {bg}")
        return

    if not fg_path.exists():
        logger.error(f"Foreground video not found: {fg}")
        return

    # Generate output path if needed
    if output is None:
        output = f"{bg_path.stem}_overlay_{fg_path.stem}.mp4"
        logger.info(f"Output path: {output}")

    # Validate output path
    output_path = Path(output)
    if output_path.exists():
        logger.warning(f"Output file already exists: {output}")
        logger.warning("It will be overwritten")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Fixed configuration based on SPEC4
    # Always use border mode with DTW and template matching
    time_mode = MatchTimeMode.BORDER
    space_method = "template"
    temporal_method = TemporalMethod.DTW
    max_keyframes = 200  # Optimal default

    if gpu:
        logger.info("GPU acceleration not yet implemented")

    # Create processor and engine
    processor = VideoProcessor()
    engine = AlignmentEngine(
        processor=processor, verbose=verbose, max_keyframes=max_keyframes
    )

    # Process the videos
    try:
        engine.process(
            bg_path=str(bg_path),
            fg_path=str(fg_path),
            output_path=output,
            time_mode=time_mode,
            space_method=space_method,
            temporal_method=temporal_method,
            skip_spatial=False,  # Always align
            trim=True,  # Always trim
            border_thickness=border,
            blend=blend,
            window=0,  # Auto-determined
        )
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    fire.Fire(main)
