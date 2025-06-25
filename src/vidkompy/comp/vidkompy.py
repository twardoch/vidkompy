#!/usr/bin/env -S uv run -s
# /// script
# dependencies = [
# "fire", "rich", "loguru", "opencv-python", "numpy", "scipy",
# "ffmpeg-python" # soundfile and scikit-image removed for MVP
# ]
# ///
# this_file: src/vidkompy/vidkompy.py

"""
Intelligent video overlay tool with automatic spatial and temporal alignment.

This tool overlays foreground videos onto background videos with smart
alignment:
- Preserves all foreground frames without retiming
- Finds optimal background frames for each foreground frame
- Supports audio-based and frame-based temporal alignment
- Automatic spatial alignment with template/feature matching
"""

import sys
from loguru import logger
from pathlib import Path

from vidkompy.comp.video import VideoProcessor
from vidkompy.comp.align import AlignmentEngine
from vidkompy.utils.enums import TimeMode


def composite_videos(
    bg: str | Path,
    fg: str | Path,
    output: str | Path | None = None,
    # engine: str = "full", # MVP: Fixed to "full" (TunnelFullSyncer)
    drift_interval: int = 10,
    margin: int = 8, # Note: Related to deferred TimeMode.BORDER
    smooth: bool = False,
    gpu: bool = False,  # Future GPU acceleration support
    window: int = 10,
    align_precision: int = 2,
    unscaled: bool = True,
    x_shift: int | None = None,
    y_shift: int | None = None,
    zero_shift: bool = False,
    verbose: bool = False,
):
    """Overlay foreground video onto background video with intelligent
    alignment. For MVP, temporal alignment engine is fixed to 'full'.

    Args:
        bg: Background video path
        fg: Foreground video path
        output: Output video path (auto-generated if not provided)
        # engine: Temporal alignment engine - 'full' or 'mask' (MVP: 'full')
        drift_interval: Frame interval for drift correction (default: 10)
        margin: Border thickness for (deferred) border matching mode (default: 8)
        smooth: Enable smooth blending at frame edges
        gpu: Enable GPU acceleration (future feature)
        window: DTW window size for temporal alignment (default: 10)
        align_precision: Spatial alignment precision level 0-4 (default: 2)
        unscaled: Prefer unscaled for spatial alignment (default: True)
        x_shift: Explicit x position of foreground onto background (disables auto-alignment)
        y_shift: Explicit y position of foreground onto background (disables auto-alignment)
        zero_shift: Force position to 0,0 and disable scaling/auto-alignment
        verbose: Enable verbose logging

    Used in:
    - vidkompy/__main__.py
    """
    # Setup logging
    logger.remove()
    log_format_verbose = (
        "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{function}</cyan> - <level>{message}</level>"
    )
    log_format_default = (
        "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<level>{message}</level>"
    )
    if verbose:
        logger.add(sys.stderr, format=log_format_verbose, level="DEBUG")
    else:
        logger.add(sys.stderr, format=log_format_default, level="INFO")

    # Log CLI options if verbose mode is enabled
    if verbose:
        logger.info("CLI options used:")
        logger.info(f"  Background video: {bg}")
        logger.info(f"  Foreground video: {fg}")
        logger.info(f"  Output path: {output}")
        logger.info(f"  Engine: full (MVP default)")
        logger.info(f"  Drift interval: {drift_interval}")
        logger.info(f"  Window: {window}")
        logger.info(f"  Margin: {margin}")
        logger.info(f"  Smooth blending: {smooth}")
        logger.info(f"  GPU acceleration: {gpu}")
        logger.info(f"  Spatial precision: {align_precision}")
        logger.info(f"  unscaled: {unscaled}")
        logger.info(f"  X shift: {x_shift}")
        logger.info(f"  Y shift: {y_shift}")
        logger.info(f"  Zero shift: {zero_shift}")
        logger.info(f"  Verbose logging: {verbose}")

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
    output_str: str
    if output is None:
        output_str = f"{bg_path.stem}_overlay_{fg_path.stem}.mp4"
        logger.info(f"Output path: {output_str}")
    else:
        output_str = str(output)

    # Validate output path
    output_path_obj = Path(output_str)
    if output_path_obj.exists():
        logger.warning(
            f"Output file {output_str} already exists. It will be overwritten."
        )

    # Ensure output directory exists
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # MVP: Engine is fixed to "full"
    engine_mode_mvp = "full"

    # Validate shift parameters
    if zero_shift and (x_shift is not None or y_shift is not None):
        logger.warning("zero_shift overrides x_shift and y_shift - setting both to 0")
        x_shift = 0
        y_shift = 0
    elif zero_shift:
        x_shift = 0
        y_shift = 0
        logger.info("Using zero_shift mode: position (0,0), no scaling")

    # Determine if we're using explicit positioning
    explicit_position = x_shift is not None or y_shift is not None or zero_shift

    # Initialize config variables with defaults
    time_mode: TimeMode = TimeMode.PRECISE
    space_method: str = "template"
    max_keyframes: int = 1  # Not used by alignment engines

    if gpu:
        logger.info("GPU acceleration not yet implemented")

    # Create processor and alignment engine
    processor = VideoProcessor()
    alignment = AlignmentEngine(
        processor=processor,
        verbose=verbose,
        max_keyframes=max_keyframes,
        engine_mode=engine_mode_mvp, # MVP: Use fixed engine mode
        drift_interval=drift_interval,
        window=window,
        spatial_precision=align_precision,
        unscaled=unscaled,
        x_shift=x_shift,
        y_shift=y_shift,
        zero_shift=zero_shift,
    )

    # Process the videos
    try:
        alignment.process(
            bg_path=str(bg_path),
            fg_path=str(fg_path),
            output_path=output_str,
            time_mode=time_mode,
            space_method=space_method,
            skip_spatial=explicit_position,  # Skip auto-alignment if explicit position
            trim=True,
            border_thickness=margin,
            blend=smooth,
            window=window,
        )
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise
