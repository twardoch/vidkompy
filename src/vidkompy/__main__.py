#!/usr/bin/env python
# this_file: src/vidkompy/__main__.py

"""Enable running vidkompy as a module with python -m vidkompy."""

from pathlib import Path

import fire


def _lazy_find_thumbnail(
    fg: str | Path,
    bg: str | Path,
    num_frames: int = 7,
    verbose: bool = False,
    precision: int = 2,
    unscaled: bool = True,
):
    """
    Find thumbnail location in background image/video.

    This function performs advanced thumbnail detection with multi-precision
    analysis to locate a foreground image within a background image or video.

    Args:
        fg: Path to foreground image or video (the thumbnail to find)
        bg: Path to background image or video (where to search)
        num_frames: Maximum number of frames to process for videos (default: 7)
        verbose: Enable verbose logging with detailed debug information
                (default: False)
        precision: Precision level 0-4 controlling speed/accuracy tradeoff
                  (default: 2)
            - 0: Ballpark (~1ms) - Ultra-fast histogram correlation only
            - 1: Coarse (~10ms) - Quick template matching
            - 2: Balanced (~25ms) - Feature + template matching combination
            - 3: Fine (~50ms) - High-quality hybrid multi-algorithm detection
            - 4: Precise (~200ms) - Maximum accuracy with sub-pixel refinement
        unscaled: If True (default), search only at 100% scale (translation
                 only). If False, perform both 100% scale and multi-scale
                 searches.

    Returns:
        None (results are displayed to console)

    Raises:
        ValueError: If precision is not between 0-4 or num_frames < 1
        Exception: If thumbnail detection fails

    """
    from .align.cli import find_thumbnail

    return find_thumbnail(fg, bg, num_frames, verbose, precision, unscaled)


def _lazy_composite_videos(
    bg: str | Path,
    fg: str | Path,
    output: str | Path | None = None,
    # engine: str = "full", # MVP: Fixed to "full"
    drift_interval: int = 10,
    margin: int = 8, # Note: Related to deferred TimeMode.BORDER
    smooth: bool = False,
    gpu: bool = False,
    window: int = 10,
    align_precision: int = 2,
    unscaled: bool = True,
    x_shift: int | None = None,
    y_shift: int | None = None,
    zero_shift: bool = False,
    verbose: bool = False,
):
    """
    Overlay foreground video onto background with intelligent alignment.

    This function performs video composition with automatic spatial and
    temporal alignment, preserving all foreground frames without retiming
    and finding optimal background frames for each foreground frame.

    Args:
        bg: Path to background video file
        fg: Path to foreground video file to overlay
        output: Output video path (auto-generated if not provided)
        # engine: Temporal alignment engine (MVP: 'full')
        #     - 'full': Direct pixel comparison with zero drift
        #     - 'mask': Content-focused comparison for letterboxed content (Deferred post-MVP)
        drift_interval: Frame interval for drift correction (default: 10)
        margin: Border thickness for (deferred) border matching mode in pixels (default: 8)
        smooth: Enable smooth blending at frame edges (default: False)
        gpu: Enable GPU acceleration - future feature (default: False)
        window: DTW window size for temporal alignment (default: 10) # Note: Review if this window is used by TunnelSyncer effectively
        align_precision: Spatial alignment precision level 0-4 (default: 2)
            - 0: Ballpark (~1ms), 1: Coarse (~10ms), 2: Balanced (~25ms)
            - 3: Fine (~50ms), 4: Precise (~200ms)
        unscaled: Prefer unscaled for spatial alignment (default: True)
        x_shift: Explicit x position of foreground onto background (disables
                auto-alignment)
        y_shift: Explicit y position of foreground onto background (disables
                auto-alignment)
        zero_shift: Force position to 0,0 and disable scaling/auto-alignment
                   (default: False)
        verbose: Enable verbose logging with detailed processing info
                (default: False)

    Returns:
        None (video is saved to output path)

    Raises:
        FileNotFoundError: If input video files don't exist
        ValueError: If engine is not 'full' or 'mask'
        Exception: If video processing fails

    Notes:
        - The Full engine is fastest with perfect confidence for standard
          videos
        - The Mask engine is ideal for letterboxed/pillarboxed content
        - Both engines achieve zero temporal drift through monotonic
          constraints
        - Processing speed is typically 5x real-time or better

    """
    from .comp.vidkompy import composite_videos

    # For MVP, engine is hardcoded to "full" in the actual composite_videos call
    return composite_videos(
        bg,
        fg,
        output,
        # engine, # Removed from call signature for MVP
        drift_interval,
        margin,
        smooth,
        gpu,
        window,
        align_precision,
        unscaled,
        x_shift,
        y_shift,
        zero_shift,
        verbose,
    )


def cli():
    """Main CLI entry point with subcommands."""

    fire.Fire(
        {
            "align": _lazy_find_thumbnail,
            "comp": _lazy_composite_videos,
        }
    )


if __name__ == "__main__":
    cli()
