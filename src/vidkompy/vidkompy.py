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
    match_time: str = "border",
    match_space: str = "precise",
    temporal_align: str = "dtw",
    skip_spatial_align: bool = False,
    trim: bool = True,
    verbose: bool = False,
    max_keyframes: int = 2000,
    border: int = 8,
    blend: bool = False,
    window: int = 0,
):
    """Overlay foreground video onto background video with intelligent alignment.

    Args:
        bg: Background video path
        fg: Foreground video path
        output: Output video path (auto-generated if not provided)
        match_time: Temporal alignment - 'border' (border matching, default), 'fast' (audio then frames), or 'precise' (frames)
        match_space: Spatial alignment - 'precise' (template) or 'fast' (feature)
        temporal_align: Temporal algorithm - 'dtw' (new default) or 'classic'
        skip_spatial_align: Skip spatial alignment, center foreground
        trim: Trim output to overlapping segments only
        verbose: Enable verbose logging
        max_keyframes: Maximum keyframes for frame-based alignment
        border: Border thickness for border matching mode (default: 8)
        blend: Enable smooth blending at frame edges
        window: Sliding window size for frame matching (0 = no window)
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

    # Validate match_time mode
    try:
        time_mode = MatchTimeMode(match_time)
    except ValueError:
        logger.error(f"Invalid match_time mode: {match_time}. Use 'border', 'fast', or 'precise'")
        return

    # Validate match_space mode
    valid_space_methods = ["precise", "template", "fast", "feature"]
    if match_space not in valid_space_methods:
        logger.error(
            f"Invalid match_space mode: {match_space}. Use one of: {', '.join(valid_space_methods)}"
        )
        return

    # Normalize space method names
    if match_space == "precise":
        match_space = "template"
    elif match_space == "fast":
        match_space = "feature"

    # Validate temporal_align
    try:
        temporal_method = TemporalMethod(temporal_align)
    except ValueError:
        # Try common aliases
        if temporal_align in ["frames", "keyframes"]:
            temporal_method = TemporalMethod.CLASSIC
        else:
            logger.error(
                f"Invalid temporal_align: {temporal_align}. Use 'dtw' or 'classic'"
            )
            return

    # Validate max_keyframes
    if max_keyframes < 10:
        logger.error(f"max_keyframes must be at least 10, got {max_keyframes}")
        return
    elif max_keyframes > 10000:
        logger.warning(
            f"max_keyframes is very high ({max_keyframes}), this may be slow"
        )

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
            space_method=match_space,
            temporal_method=temporal_method,
            skip_spatial=skip_spatial_align,
            trim=trim,
            border_thickness=border,
            blend=blend,
            window=window,
        )
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    fire.Fire(main)
