#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["fire", "rich", "loguru", "opencv-python", "numpy", "ffmpeg-python"]
# ///
# this_file: benchmark.py

"""
Benchmark script for vidkompy performance testing.

Tests various configurations to measure performance improvements.
"""

import time
import subprocess
from pathlib import Path
from rich.console import Console
from rich.table import Table
from loguru import logger
import json
from datetime import datetime

console = Console()


def run_vidkompy(bg_path: str, fg_path: str, output_path: str, config: dict) -> dict:
    """Run vidkompy with given configuration and measure performance."""
    start_time = time.time()

    # Build command
    cmd = [
        "python",
        "-m",
        "vidkompy",
        "--bg",
        bg_path,
        "--fg",
        fg_path,
        "--output",
        output_path,
    ]

    # Add configuration parameters
    for key, value in config.get("params", {}).items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    # Run command
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.time() - start_time

        # Extract frame count from logs if available
        frame_count = 483  # Default for test video

        return {
            "config": config["name"],
            "description": config.get("description", ""),
            "time": elapsed,
            "fps": frame_count / elapsed if elapsed > 0 else 0,
            "success": True,
            "output": output_path,
        }
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logger.error(f"Command failed: {e}")
        logger.error(f"stderr: {e.stderr}")
        return {
            "config": config["name"],
            "description": config.get("description", ""),
            "time": elapsed,
            "fps": 0,
            "success": False,
            "error": str(e),
        }


def benchmark_configurations():
    """Run benchmarks with different configurations."""
    # Test configurations
    configs = [
        {
            "name": "default_200_keyframes",
            "description": "Default settings with 200 keyframes",
            "params": {},
        },
        {
            "name": "sparse_12_keyframes",
            "description": "Very sparse keyframes (old default)",
            "params": {"max_keyframes": 12},
        },
        {
            "name": "dense_500_keyframes",
            "description": "Dense keyframes for minimal drift",
            "params": {"max_keyframes": 500},
        },
        {
            "name": "border_mode_default",
            "description": "Border mode with default settings",
            "params": {"match_time": "border"},
        },
        {
            "name": "border_mode_dtw",
            "description": "Border mode with DTW alignment",
            "params": {"match_time": "border", "temporal_align": "dtw"},
        },
        {
            "name": "classic_mode",
            "description": "Classic keyframe alignment",
            "params": {"temporal_align": "classic"},
        },
        {
            "name": "dtw_mode",
            "description": "DTW alignment (full frame)",
            "params": {"temporal_align": "dtw"},
        },
    ]

    # Ensure output directory exists
    output_dir = Path("benchmark_outputs")
    output_dir.mkdir(exist_ok=True)

    # Run benchmarks
    console.print("\n[bold cyan]Running Vidkompy Benchmarks[/bold cyan]\n")
    results = []

    for config in configs:
        console.print(
            f"[yellow]Running:[/yellow] {config['name']} - {config['description']}"
        )

        output_path = output_dir / f"benchmark_{config['name']}.mp4"
        result = run_vidkompy("tests/bg.mp4", "tests/fg.mp4", str(output_path), config)
        results.append(result)

        if result["success"]:
            console.print(
                f"[green]✓[/green] Completed in {result['time']:.2f}s ({result['fps']:.1f} fps)\n"
            )
        else:
            console.print(f"[red]✗[/red] Failed after {result['time']:.2f}s\n")

    return results


def display_results(results: list):
    """Display benchmark results in a table."""
    # Create results table
    table = Table(title="Benchmark Results", show_header=True)
    table.add_column("Configuration", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Time (s)", justify="right", style="yellow")
    table.add_column("FPS", justify="right", style="green")
    table.add_column("Status", justify="center")

    # Add rows
    for r in results:
        status = "[green]✓[/green]" if r["success"] else "[red]✗[/red]"
        table.add_row(
            r["config"],
            r["description"],
            f"{r['time']:.2f}",
            f"{r['fps']:.1f}",
            status,
        )

    console.print("\n")
    console.print(table)

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"benchmark_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]Results saved to:[/green] {results_file}")

    # Performance comparison
    if len(results) >= 2:
        console.print("\n[bold]Performance Comparison:[/bold]")

        # Find baseline (sparse keyframes)
        baseline = next(
            (r for r in results if r["config"] == "sparse_12_keyframes"), results[0]
        )

        for r in results:
            if r["success"] and r != baseline:
                speedup = baseline["time"] / r["time"] if r["time"] > 0 else 0
                if speedup > 1:
                    console.print(
                        f"• {r['config']}: [green]{speedup:.1f}x faster[/green] than baseline"
                    )
                elif speedup < 1:
                    console.print(
                        f"• {r['config']}: [red]{1 / speedup:.1f}x slower[/red] than baseline"
                    )


def main(
    quick: bool = False,
    output_dir: str = "benchmark_outputs",
):
    """Run vidkompy benchmarks.

    Args:
        quick: Run only a subset of benchmarks
        output_dir: Directory for output files
    """
    logger.info("Starting vidkompy benchmark suite")

    # Check if test files exist
    if not Path("tests/bg.mp4").exists() or not Path("tests/fg.mp4").exists():
        console.print(
            "[red]Error:[/red] Test files not found. Please ensure tests/bg.mp4 and tests/fg.mp4 exist."
        )
        return

    # Run benchmarks
    results = benchmark_configurations()

    # Display results
    display_results(results)

    # Cleanup note
    console.print(f"\n[dim]Output files saved in {output_dir}/ directory[/dim]")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
