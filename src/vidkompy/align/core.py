#!/usr/bin/env python3
# this_file: src/vidkompy/align/comp.py

"""
Core thumbnail finder orchestrator.

This module contains the main ThumbnailFinder class that coordinates
between all the other components to provide the high-level interface.
"""

from pathlib import Path

import numpy as np

from .result_types import (
    ThumbnailResult,
    AnalysisData,
    MatchResult,
)
from .frame_extractor import FrameExtractor
from .precision import PrecisionAnalyzer
from .algorithms import TemplateMatchingAlgorithm
from .display import ResultDisplayer


class ThumbnailFinder:
    """
    Main thumbnail finder orchestrator.

    This class coordinates between frame extraction, precision analysis,
    algorithms, and display to provide a high-level interface for
    thumbnail detection.
    """

    def __init__(self):
        """Initialize the thumbnail finder with component instances."""
        self.frame_extractor = FrameExtractor()
        self.precision_analyzer = PrecisionAnalyzer()
        self.template_matcher = TemplateMatchingAlgorithm()
        self.displayer = ResultDisplayer()

    def find_thumbnail(
        self,
        fg: str | Path,
        bg: str | Path,
        num_frames: int = 7,
        verbose: bool = False,
        precision: int = 2,
        unity_scale: bool = True,
    ) -> ThumbnailResult:
        """
        Find thumbnail in background image/video.

        Args:
            fg: Path to foreground image or video
            bg: Path to background image or video
            num_frames: Maximum number of frames to process
            verbose: Enable verbose logging
            precision: Precision level 0-4
            unity_scale: If True, only search at 100% scale. If False,
                        search both at 100% scale and multi-scale.

        Returns:
            ThumbnailResult with detection results
        """
        fg_path = Path(fg)
        bg_path = Path(bg)

        # Display header
        self.displayer.display_header(fg_path, bg_path, num_frames, precision)

        # Extract frames
        fg_extraction = self.frame_extractor.extract_frames(
            fg_path, num_frames, verbose
        )
        bg_extraction = self.frame_extractor.extract_frames(
            bg_path, num_frames, verbose
        )

        self.displayer.display_extraction_progress(
            fg_extraction.frame_count, bg_extraction.frame_count
        )

        # Perform thumbnail detection
        with self.displayer.create_search_progress() as progress:
            task = progress.add_task("Searching for thumbnails...", total=100)

            result = self._find_thumbnail_in_frames(
                fg_extraction.frames,
                bg_extraction.frames,
                precision,
                unity_scale,
                progress,
                task,
            )

            progress.update(task, completed=100)

        # Create complete result object
        thumbnail_result = ThumbnailResult(
            confidence=result[0],
            scale_fg_to_thumb=result[1],
            x_thumb_in_bg=int(result[2]),
            y_thumb_in_bg=int(result[3]),
            scale_bg_to_fg=result[4],
            x_fg_in_scaled_bg=int(result[5]),
            y_fg_in_scaled_bg=int(result[6]),
            analysis_data=result[7],
            fg_size=fg_extraction.original_size,
            bg_size=bg_extraction.original_size,
            thumbnail_size=self._calculate_thumbnail_size(
                fg_extraction.original_size, result[1]
            ),
            upscaled_bg_size=self._calculate_upscaled_bg_size(
                bg_extraction.original_size, result[4]
            ),
        )

        # Display results
        if hasattr(thumbnail_result.analysis_data, "precision_analysis"):
            self.displayer.display_precision_analysis(
                thumbnail_result.analysis_data.precision_analysis
            )

        self.displayer.display_main_results_table(thumbnail_result)
        self.displayer.display_summary(thumbnail_result, fg_path, bg_path)
        self.displayer.display_alternative_analysis(thumbnail_result.analysis_data)

        return thumbnail_result

    def _find_thumbnail_in_frames(
        self,
        fg_frames: list[np.ndarray],
        bg_frames: list[np.ndarray],
        precision: int,
        unity_scale: bool,
        progress=None,
        task=None,
    ) -> tuple:
        """
        Core thumbnail detection logic.

        Args:
            fg_frames: List of foreground frames
            bg_frames: List of background frames
            precision: Precision level
            unity_scale: Unity scale mode flag
            progress: Optional progress bar
            task: Optional progress task

        Returns:
            Tuple of detection results
        """
        if not fg_frames or not bg_frames:
            return (0.0, 0.0, 0, 0, 100.0, 0, 0, AnalysisData())

        # Use first frames for analysis
        first_fg = fg_frames[0]
        first_bg = bg_frames[0]

        # Convert to grayscale for analysis
        if len(first_fg.shape) == 3:
            first_fg_gray = self.frame_extractor.preprocess_frame(
                first_fg, grayscale=True
            )
        else:
            first_fg_gray = first_fg

        if len(first_bg.shape) == 3:
            first_bg_gray = self.frame_extractor.preprocess_frame(
                first_bg, grayscale=True
            )
        else:
            first_bg_gray = first_bg

        # Perform precision analysis
        precision_results = self.precision_analyzer.progressive_analysis(
            first_fg_gray, first_bg_gray, precision
        )

        if progress and task:
            progress.update(task, advance=50)

        # Get best result from precision analysis
        best_precision_result = max(precision_results, key=lambda r: r.confidence)

        # Create main match result
        main_result = MatchResult(
            confidence=best_precision_result.confidence,
            x=best_precision_result.x,
            y=best_precision_result.y,
            scale=best_precision_result.scale,
            method=best_precision_result.method,
        )

        # Handle unity scale logic
        analysis_data = AnalysisData(
            precision_level=precision,
            unity_scale_preference_active=unity_scale,
            precision_analysis=precision_results,
        )

        if unity_scale:
            # Unity scale mode: only 100% scale search
            unity_result = self.template_matcher.match_template(
                first_fg_gray, first_bg_gray, scale=1.0, method="unity"
            )
            analysis_data.unity_scale_result = unity_result
            main_result = unity_result
        else:
            # Multi-scale mode: both unity and scaled searches
            unity_result = self.template_matcher.match_template(
                first_fg_gray, first_bg_gray, scale=1.0, method="unity"
            )
            analysis_data.unity_scale_result = unity_result
            analysis_data.scaled_result = main_result

        if progress and task:
            progress.update(task, advance=30)

        # Calculate final transformation parameters
        confidence = main_result.confidence
        scale_fg_to_thumb = main_result.scale * 100.0
        x_thumb_in_bg = main_result.x
        y_thumb_in_bg = main_result.y

        # Calculate reverse transformation
        scale_bg_to_fg = (100.0 / main_result.scale) if main_result.scale > 0 else 100.0
        x_fg_in_scaled_bg = (
            -int(main_result.x / main_result.scale) if main_result.scale > 0 else 0
        )
        y_fg_in_scaled_bg = (
            -int(main_result.y / main_result.scale) if main_result.scale > 0 else 0
        )

        if progress and task:
            progress.update(task, advance=20)

        return (
            confidence,
            scale_fg_to_thumb,
            x_thumb_in_bg,
            y_thumb_in_bg,
            scale_bg_to_fg,
            x_fg_in_scaled_bg,
            y_fg_in_scaled_bg,
            analysis_data,
        )

    def _calculate_thumbnail_size(
        self, fg_size: tuple[int, int], scale_percent: float
    ) -> tuple[int, int]:
        """Calculate thumbnail size from foreground size and scale."""
        scale_factor = scale_percent / 100.0
        return (int(fg_size[0] * scale_factor), int(fg_size[1] * scale_factor))

    def _calculate_upscaled_bg_size(
        self, bg_size: tuple[int, int], scale_percent: float
    ) -> tuple[int, int]:
        """Calculate upscaled background size."""
        scale_factor = scale_percent / 100.0
        return (int(bg_size[0] * scale_factor), int(bg_size[1] * scale_factor))

    def validate_inputs(self, fg_path: Path, bg_path: Path) -> bool:
        """
        Validate input file paths.

        Args:
            fg_path: Foreground file path
            bg_path: Background file path

        Returns:
            True if inputs are valid

        Raises:
            FileNotFoundError: If files don't exist
            ValueError: If files are invalid
        """
        if not fg_path.exists():
            msg = f"Foreground file not found: {fg_path}"
            raise FileNotFoundError(msg)

        if not bg_path.exists():
            msg = f"Background file not found: {bg_path}"
            raise FileNotFoundError(msg)

        # Additional validation could be added here
        return True
