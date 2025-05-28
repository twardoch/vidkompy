#!/usr/bin/env python3
# this_file: src/vidkompy/align/core.py

"""
Core thumbnail finder orchestrator.

This module contains the main ThumbnailFinder class that coordinates
between all the other components to provide the high-level interface.

"""

from pathlib import Path
import logging

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
from src.vidkompy.utils.image import ensure_gray


class ThumbnailFinder:
    """
    Main thumbnail finder orchestrator.

    This class coordinates between frame extraction, precision analysis,
    algorithms, and display to provide a high-level interface for
    thumbnail detection.

    Used in:
    - vidkompy/align/__init__.py
    - vidkompy/align/cli.py
    - vidkompy/comp/alignment_engine.py
    """

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize the thumbnail finder with component instances."""
        self.frame_extractor = FrameExtractor()
        self.precision_analyzer = PrecisionAnalyzer()
        self.template_matcher = TemplateMatchingAlgorithm()
        self.displayer = ResultDisplayer()
        self.logger = logger or logging.getLogger(__name__)

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

        Used in:
        - vidkompy/align/cli.py
        - vidkompy/comp/alignment_engine.py
        """
        fg_path = Path(fg)
        bg_path = Path(bg)

        # Validate inputs
        self.validate_inputs(fg_path, bg_path)

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
        Core thumbnail detection logic using modular helper methods.

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

        # Step 1: Prepare grayscale frames for analysis
        try:
            first_fg_gray, first_bg_gray = self._prepare_gray_frames(
                fg_frames, bg_frames
            )
        except ValueError as e:
            self.logger.error(f"Frame preparation failed: {e}")
            return (0.0, 0.0, 0, 0, 100.0, 0, 0, AnalysisData())

        # Step 2: Run precision analysis
        precision_results = self._run_precision_analysis(
            first_fg_gray, first_bg_gray, precision
        )

        if progress and task:
            progress.update(task, advance=50)

        # Step 3: Select main result based on analysis and unity scale preference
        main_result, analysis_data = self._select_main_result(
            precision_results, unity_scale, first_fg_gray, first_bg_gray
        )

        if progress and task:
            progress.update(task, advance=30)

        # Step 4: Build final thumbnail result
        result = self._build_thumbnail_result(main_result, analysis_data)

        if progress and task:
            progress.update(task, advance=20)

        return result

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

    def _prepare_gray_frames(
        self, fg_frames: list[np.ndarray], bg_frames: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare grayscale versions of the first frames for analysis.

        Args:
            fg_frames: List of foreground frames
            bg_frames: List of background frames

        Returns:
            Tuple of (first_fg_gray, first_bg_gray)

        """
        if not fg_frames or not bg_frames:
            msg = "Frame lists cannot be empty"
            raise ValueError(msg)

        first_fg = fg_frames[0]
        first_bg = bg_frames[0]

        # Convert to grayscale using utils
        first_fg_gray = ensure_gray(first_fg)
        first_bg_gray = ensure_gray(first_bg)

        return first_fg_gray, first_bg_gray

    def _run_precision_analysis(
        self, fg_gray: np.ndarray, bg_gray: np.ndarray, precision: int
    ) -> list:
        """
        Run precision analysis on grayscale frames.

        Args:
            fg_gray: Grayscale foreground frame
            bg_gray: Grayscale background frame
            precision: Precision level

        Returns:
            List of precision analysis results

        """
        return self.precision_analyzer.progressive_analysis(fg_gray, bg_gray, precision)

    def _select_main_result(
        self,
        precision_results: list,
        unity_scale: bool,
        fg_gray: np.ndarray,
        bg_gray: np.ndarray,
    ) -> tuple[MatchResult, AnalysisData]:
        """
        Select the main result based on precision analysis and unity scale preference.

        Args:
            precision_results: Results from precision analysis
            unity_scale: Unity scale mode flag
            fg_gray: Grayscale foreground frame
            bg_gray: Grayscale background frame

        Returns:
            Tuple of (main_result, analysis_data)

        """
        # Get best result from precision analysis
        best_precision_result = max(precision_results, key=lambda r: r.confidence)

        # Create analysis data container
        analysis_data = AnalysisData(
            precision_level=precision_results[0].method
            if precision_results
            else "unknown",
            unity_scale_preference_active=unity_scale,
            precision_analysis=precision_results,
        )

        if unity_scale:
            # Unity scale mode: prefer 100% scale but fallback to best precision result
            unity_result = self.template_matcher.match_template(
                fg_gray, bg_gray, scale=1.0, method="unity"
            )
            analysis_data.unity_scale_result = unity_result

            # Use unity result only if it's better than precision analysis
            if unity_result.confidence > best_precision_result.confidence:
                main_result = unity_result
            else:
                # Keep the best precision result if unity scale fails
                main_result = MatchResult(
                    confidence=best_precision_result.confidence,
                    x=best_precision_result.x,
                    y=best_precision_result.y,
                    scale=best_precision_result.scale,
                    method=f"precision_fallback_{best_precision_result.method}",
                )
        else:
            # Multi-scale mode: both unity and scaled searches
            unity_result = self.template_matcher.match_template(
                fg_gray, bg_gray, scale=1.0, method="unity"
            )
            analysis_data.unity_scale_result = unity_result
            analysis_data.scaled_result = best_precision_result

            # Use the best of both approaches
            main_result = best_precision_result

        return main_result, analysis_data

    def _build_thumbnail_result(
        self, main_result: MatchResult, analysis_data: AnalysisData
    ) -> tuple:
        """
        Build the final thumbnail result tuple from the main result.

        Args:
            main_result: The selected main match result
            analysis_data: Analysis data container

        Returns:
            Tuple of final transformation parameters

        """
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
