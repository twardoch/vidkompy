#!/usr/bin/env python3
# this_file: src/vidkompy/align/cli.py

"""
Command-line interface for thumbnail detection.

This module provides the Fire-based CLI interface for the thumbnail finder,
handling parameter validation and entry point logic.
"""

from pathlib import Path
import sys

import fire
from loguru import logger

from .core import ThumbnailFinder


def find_thumbnail(
    fg: str | Path,
    bg: str | Path,
    num_frames: int = 7,
    verbose: bool = False,
    precision: int = 2,
    unity_scale: bool = True,
):
    """
    Main entry point for the thumbnail finder.

    Args:
        fg: Path to foreground image or video
        bg: Path to background image or video
        num_frames: Maximum number of frames to process
        verbose: Enable verbose logging
        precision: Precision level 0-4 (0=ballpark ~1ms, 1=coarse ~10ms,
                  2=balanced ~25ms, 3=fine ~50ms, 4=precise ~200ms)
        unity_scale: If True (default), performs a search ONLY for the
                    foreground at 100% scale (translation only). If False,
                    performs both a 100% scale search and a multi-scale search,
                    presenting both results.

    Used in:
    - vidkompy/__main__.py
    - vidkompy/align/__init__.py
    """
    # Configure logging level
    if verbose:
        logger.add(sys.stderr, format=log_format_verbose, level="DEBUG")
    else:
        logger.add(sys.stderr, format=log_format_default, level="INFO")

    # Validate parameters
    if not 0 <= precision <= 4:
        msg = f"Precision must be between 0 and 4, got {precision}"
        raise ValueError(msg)

    if num_frames < 1:
        msg = f"num_frames must be at least 1, got {num_frames}"
        raise ValueError(msg)

    # Create and run thumbnail finder
    align = ThumbnailFinder()

    try:
        align.find_thumbnail(
            fg=fg,
            bg=bg,
            num_frames=num_frames,
            verbose=verbose,
            precision=precision,
            unity_scale=unity_scale,
        )
        # Fire CLI: Don't return the result object to avoid help display
        # The result is already displayed by the find_thumbnail method
        return None

    except Exception as e:
        logger.error(f"Thumbnail detection failed: {e}")
        raise


def main():
    """Main entry point for Fire CLI."""
    fire.Fire(find_thumbnail)


if __name__ == "__main__":
    main()
