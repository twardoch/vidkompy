#!/usr/bin/env python3
# this_file: src/vidkompy/align/display.py

"""
Display and output formatting for thumbnail detection results.

This module handles all Rich console output, table formatting,
progress bars, and result presentation.

"""

from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.text import Text

from vidkompy.align.data_types import (
    ThumbnailResult,
    PrecisionAnalysisResult,
    PrecisionLevel,
    AnalysisData,
)


class ResultDisplayer:
    """
    Handles formatting and display of thumbnail detection results.

    This class provides methods to display results in various formats
    using Rich console output with tables, progress bars, and styling.

    Used in:
    - vidkompy/align/__init__.py
    - vidkompy/align/core.py
    """

    def __init__(self, console: Console | None = None):
        """
        Initialize the result displayer.

        Args:
            console: Optional Rich console instance

        """
        self.console = console or Console()

    def display_header(
        self, fg_path: Path, bg_path: Path, max_frames: int, precision_level: int
    ):
        """
        Display the header information for thumbnail detection.

        Args:
            fg_path: Foreground file path
            bg_path: Background file path
            max_frames: Maximum frames to process
            precision_level: Precision level being used

        Used in:
        - vidkompy/align/core.py
        """
        precision_desc = self._get_precision_description(precision_level)

        self.console.print("\n[bold cyan]Thumbnail Finder[/bold cyan]")
        self.console.print(f"Foreground: {fg_path.name}")
        self.console.print(f"Background: {bg_path.name}")
        self.console.print(f"Max frames: {max_frames}")
        self.console.print(f"Precision Level: {precision_level} - {precision_desc}")

    def display_extraction_progress(self, fg_count: int, bg_count: int):
        """
        Display frame extraction results.

        Args:
            fg_count: Number of foreground frames extracted
            bg_count: Number of background frames extracted

        Used in:
        - vidkompy/align/core.py
        """
        self.console.print("\nExtracting frames...")
        self.console.print(f"Extracted {fg_count} foreground frames")
        self.console.print(f"Extracted {bg_count} background frames")

    def create_search_progress(self) -> Progress:
        """
        Create a progress bar for thumbnail search.

        Returns:
            Rich Progress instance

        Used in:
        - vidkompy/align/core.py
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            console=self.console,
        )

    def display_precision_analysis(
        self, precision_results: list[PrecisionAnalysisResult]
    ):
        """
        Display precision analysis results.

        Args:
            precision_results: List of results from each precision level

        Used in:
        - vidkompy/align/core.py
        """
        if not precision_results:
            return

        self.console.print("\n[bold blue]Precision Analysis (first frame):[/bold blue]")
        self.console.print()

        max_level = max(r.level.value for r in precision_results)
        self.console.print(f"● Precision Level {max_level}")

        for result in precision_results:
            level_desc = result.level.description

            if result.level == PrecisionLevel.BALLPARK:
                self.console.print(
                    f"  Level {result.level.value} ({level_desc}): "
                    f"scale ≈ {result.scale * 100:.1f}%, "
                    f"confidence = {result.confidence:.3f}"
                )
            else:
                self.console.print(
                    f"  Level {result.level.value} ({level_desc}): "
                    f"scale = {result.scale * 100:.2f}%, "
                    f"pos = ({result.x}, {result.y}), "
                    f"confidence = {result.confidence:.3f}"
                )

    def display_main_results_table(self, result: ThumbnailResult):
        """
        Display the main results table.

        Args:
            result: Complete thumbnail detection result
        """
        table = Table(title="Thumbnail Detection Results")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        table.add_column("Unit", style="green")

        # Add rows to the table
        table.add_row("Confidence", f"{result.confidence * 100:.2f}", "%")
        table.add_row(
            "FG file",
            Path(result.analysis_data.fg_file).name
            if hasattr(result.analysis_data, "fg_file")
            else "N/A",
            "",
        )
        table.add_row("FG original size", f"{result.fg_width}×{result.fg_height}", "px")
        table.add_row(
            "BG file",
            Path(result.analysis_data.bg_file).name
            if hasattr(result.analysis_data, "bg_file")
            else "N/A",
            "",
        )
        table.add_row("BG original size", f"{result.bg_width}×{result.bg_height}", "px")
        table.add_row("Scale (FG → thumbnail)", f"{result.scale_fg_to_thumb:.2f}", "%")
        table.add_row(
            "Thumbnail size",
            f"{result.thumbnail_width}×{result.thumbnail_height}",
            "px",
        )
        table.add_row("X shift (thumbnail in BG)", str(result.x_thumb_in_bg), "px")
        table.add_row("Y shift (thumbnail in BG)", str(result.y_thumb_in_bg), "px")
        table.add_row("Scale (BG → FG size)", f"{result.scale_bg_to_fg:.2f}", "%")
        table.add_row(
            "Upscaled BG size",
            f"{result.upscaled_bg_size[0]}×{result.upscaled_bg_size[1]}",
            "px",
        )
        table.add_row(
            "X shift (FG on upscaled BG)", str(result.x_fg_in_scaled_bg), "px"
        )
        table.add_row(
            "Y shift (FG on upscaled BG)", str(result.y_fg_in_scaled_bg), "px"
        )

        self.console.print()
        self.console.print(table)

    def display_summary(self, result: ThumbnailResult, fg_path: Path, bg_path: Path):
        """
        Display a concise summary of results.

        Args:
            result: Complete thumbnail detection result
            fg_path: Foreground file path
            bg_path: Background file path
        """
        self.console.print("\n[bold green]Summary:[/bold green]")
        self.console.print(f"FG file: {fg_path.name}")
        self.console.print(f"BG file: {bg_path.name}")
        self.console.print(f"Confidence: {result.confidence * 100:.2f}%")
        self.console.print(f"FG size: {result.fg_width}×{result.fg_height} px")
        self.console.print(f"BG size: {result.bg_width}×{result.bg_height} px")
        self.console.print(
            f"Scale down: {result.scale_fg_to_thumb:.2f}% → "
            f"{result.thumbnail_width}×{result.thumbnail_height} px"
        )
        self.console.print(
            f"Position: ({result.x_thumb_in_bg}, {result.y_thumb_in_bg}) px"
        )
        self.console.print(
            f"Scale up: {result.scale_bg_to_fg:.2f}% → "
            f"{result.upscaled_bg_size[0]}×{result.upscaled_bg_size[1]} px"
        )
        self.console.print(
            f"Reverse position: ({result.x_fg_in_scaled_bg}, {result.y_fg_in_scaled_bg}) px"
        )

    def display_alternative_analysis(self, analysis_data: AnalysisData):
        """
        Display alternative analysis results.

        Args:
            analysis_data: Analysis data with alternative results
        """
        self.console.print("\n[bold blue]Alternative Analysis:[/bold blue]")

        if analysis_data.unscaled_result:
            no = analysis_data.unscaled_result
            self.console.print(
                f"  unscaled (100%) option: confidence={no.confidence:.3f}, "
                f"position=({no.x}, {no.y})"
            )

        if analysis_data.scaled_result:
            scaled = analysis_data.scaled_result
            self.console.print(
                f"  Scaled option: confidence={scaled.confidence:.3f}, "
                f"scale={scaled.scale * 100:.2f}%, position=({scaled.x}, {scaled.y})"
            )

        # Display preference mode
        if analysis_data.unscaled_preference_active:
            self.console.print("  Preference mode: unscaled preferred")
        else:
            self.console.print("  Preference mode: Multi-scale analysis")

        # Display result counts
        total = analysis_data.total_results
        unscaled_count = analysis_data.unscaled_count
        self.console.print(
            f"  Total results analyzed: {total} ({unscaled_count} near 100% scale)"
        )

    def display_verbose_info(self, message: str):
        """
        Display verbose information if verbose mode is enabled.

        Args:
            message: Message to display
        """
        self.console.print(f"[dim]INFO: {message}[/dim]")

    def display_error(self, message: str):
        """
        Display an error message.

        Args:
            message: Error message to display
        """
        self.console.print(f"[bold red]ERROR: {message}[/bold red]")

    def display_warning(self, message: str):
        """
        Display a warning message.

        Args:
            message: Warning message to display
        """
        self.console.print(f"[bold yellow]WARNING: {message}[/bold yellow]")

    def _get_precision_description(self, precision_level: int) -> str:
        """
        Get description for precision level.

        Args:
            precision_level: Precision level (0-4)

        Returns:
            Human-readable description
        """
        try:
            level = PrecisionLevel(precision_level)
            return f"{level.description} ({level.timing_estimate})"
        except ValueError:
            return "Unknown"

    def create_compact_table(self, data: dict[str, str]) -> Table:
        """
        Create a compact table for displaying key-value data.

        Args:
            data: Dictionary of key-value pairs

        Returns:
            Rich Table instance
        """
        table = Table(show_header=False, box=None)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="white")

        for key, value in data.items():
            table.add_row(key, value)

        return table

    def format_confidence(self, confidence: float) -> Text:
        """
        Format confidence score with color coding.

        Args:
            confidence: Confidence score (0.0 to 1.0)

        Returns:
            Rich Text with appropriate styling
        """
        percentage = confidence * 100

        if percentage >= 90:
            color = "green"
        elif percentage >= 70:
            color = "yellow"
        elif percentage >= 50:
            color = "orange"
        else:
            color = "red"

        return Text(f"{percentage:.2f}%", style=color)

    def format_position(self, x: int, y: int) -> str:
        """
        Format position coordinates.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Formatted position string
        """
        return f"({x}, {y})"

    def format_size(self, width: int, height: int) -> str:
        """
        Format size dimensions.

        Args:
            width: Width in pixels
            height: Height in pixels

        Returns:
            Formatted size string
        """
        return f"{width}×{height}"

    def format_scale(self, scale: float) -> str:
        """
        Format scale factor as percentage.

        Args:
            scale: Scale factor

        Returns:
            Formatted scale string
        """
        return f"{scale * 100:.2f}%"
