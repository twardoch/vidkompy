# `vidkompy`

[![PyPI](https://img.shields.io/pypi/v/vidkompy.svg)](https://pypi.org/project/vidkompy/) [![License](https://img.shields.io/github/license/twardoch/vidkompy.svg)](LICENSE)

**Intelligent Video Overlay and Synchronization**

`vidkompy` is a powerful Python-based tool designed for intelligently overlaying a foreground video onto a background video with exceptional precision and automatic alignment. It expertly handles common discrepancies such as differences in resolution, frame rate, duration, and audio, prioritizing content integrity and synchronization accuracy.

The core philosophy of `vidkompy` is to treat the **foreground video as the definitive source of quality and timing**. All its frames are preserved without modification or re-timing. The background video is dynamically adapted—stretched, retimed, and selectively sampled—to synchronize perfectly with every frame of the foreground content, ensuring a seamless and coherent final output.

## Who is `vidkompy` for?

`vidkompy` is for:

*   **Video editors and content creators** who need to precisely overlay video clips, such as picture-in-picture effects, screen recordings onto presentations, or aligning different camera angles of the same event.
*   **Archivists and media professionals** working with historical footage or multiple video sources that need careful synchronization.
*   **Developers** looking for a robust library to integrate automated video alignment and composition into their applications.
*   **Anyone** who needs to combine videos accurately without manual frame-by-frame adjustments.

## Why is `vidkompy` useful?

`vidkompy` saves time and effort by automating complex video alignment tasks. Its key benefits include:

*   **High Precision:** Achieves sub-second temporal accuracy and precise spatial positioning.
*   **Content Integrity:** Prioritizes the foreground video, ensuring no frames are dropped or unnecessarily altered.
*   **Automatic Alignment:**
    *   **Spatial:** Intelligently detects the optimal x/y offset to position the foreground video, even with different croppings.
    *   **Temporal:** Aligns videos with different start times, durations, and frame rates, eliminating drift.
*   **Resilience:** Handles variations in video properties like resolution and aspect ratio.
*   **Smart Audio Handling:** Uses foreground audio by default, ensuring it remains synchronized with the composed video.
*   **Flexible:** Offers both a command-line interface for quick use and a Python API for integration.
*   **Performance:** Leverages optimized algorithms and efficient video processing techniques.

Key Features:
*   **Foreground-First Principle:** Every frame of the foreground video is included in the output, preserving its original timing and quality.
*   **Drift-Free Alignment:** Employs advanced algorithms for globally optimal, monotonic temporal alignment.
*   **Specialized Modes:** Supports different alignment engines (e.g., `mask` for letterboxed content) and options like smooth blending.
*   **Advanced Thumbnail Detection:** A standalone module (`align`) can find an image (or video frame) within another image or video with multiple precision levels and algorithms.

## Prerequisites

Before using `vidkompy`, you must have **FFmpeg** installed on your system and accessible in your system's `PATH`. `vidkompy` relies on FFmpeg for all underlying video and audio processing tasks.

You can download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html).

## Installation

`vidkompy` is a Python package. It is recommended to install it from the official repository to get the latest version.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/twardoch/vidkompy.git
    cd vidkompy
    ```

2.  **Install using `uv` (recommended) or `pip`:**
    ```bash
    # Using uv
    uv pip install .

    # Or using pip
    pip install .
    ```
    This will install `vidkompy` and its dependencies.

## How to Use `vidkompy`

`vidkompy` can be used via its command-line interface (CLI) or programmatically in your Python scripts.

### Command-Line Interface (CLI)

The CLI is accessed by running `python -m vidkompy`. It has two main subcommands: `comp` (for video composition) and `align` (for thumbnail detection).

#### Video Composition (`comp`)

This command overlays a foreground video onto a background video.

**Basic Usage:**
```bash
python -m vidkompy comp --bg background.mp4 --fg foreground.mp4
```
This will create an output file like `background_overlay_foreground.mp4` in the current directory.

**Common Options:**
*   `--output <filepath>`: Specify a custom output file path.
    ```bash
    python -m vidkompy comp --bg bg.mp4 --fg fg.mp4 --output result.mp4
    ```
*   `--engine <enginename>`: Choose the temporal alignment engine. (Currently defaults to `full`, `mask` engine is also available).
    ```bash
    # For letterboxed/pillarboxed content
    python -m vidkompy comp --bg bg.mp4 --fg fg.mp4 --engine mask
    ```
*   `--align_precision <0-4>`: Set spatial alignment precision (default: 2). Higher is more accurate but slower.
    ```bash
    python -m vidkompy comp --bg bg.mp4 --fg fg.mp4 --align_precision 4
    ```
*   `--unscaled <true|false>`: For spatial alignment, `true` (default) prefers unscaled matches. `false` allows scaled matches.
*   `--verbose`: Enable detailed logging output.

**Example with more options:**
```bash
python -m vidkompy comp --bg main_presentation.mp4 --fg speaker_inset.mp4 --output final_video.mp4 --align_precision 3 --verbose
```

#### Thumbnail Detection (`align`)

This command finds a foreground image (or video frame) within a background image or video.

**Basic Usage:**
```bash
python -m vidkompy align thumbnail_image.jpg main_video.mp4
```

**Common Options:**
*   `--precision <0-4>`: Set detection precision (default: 2).
    *   `0`: Ballpark (~1ms) - Ultra-fast.
    *   `1`: Coarse (~10ms) - Quick.
    *   `2`: Balanced (~25ms) - Default.
    *   `3`: Fine (~50ms) - High quality.
    *   `4`: Precise (~200ms) - Maximum accuracy.
*   `--num_frames <number>`: Max frames to process if inputs are videos (default: 7).
*   `--unscaled <true|false>`: `true` (default) searches only at 100% scale. `false` searches multi-scale.
*   `--verbose`: Enable detailed output and debug logging.

**Example:**
```bash
python -m vidkompy align company_logo.png corporate_video.mp4 --precision 3 --verbose
```

#### Getting Help

To see all available commands and their options:
```bash
# Main help
python -m vidkompy --help

# Help for video composition
python -m vidkompy comp --help

# Help for thumbnail detection
python -m vidkompy align --help
```

### Programmatic Usage

You can integrate `vidkompy`'s functionalities directly into your Python projects.

#### Video Composition

To overlay videos programmatically, use the `composite_videos` function.

```python
from vidkompy.comp.vidkompy import composite_videos
from pathlib import Path

# Define paths to your video files
background_video = Path("path/to/your/background.mp4")
foreground_video = Path("path/to/your/foreground.mp4")
output_video = Path("path/to/your/composed_output.mp4")

try:
    composite_videos(
        bg=str(background_video),
        fg=str(foreground_video),
        output=str(output_video),
        # engine="full",  # Default, 'mask' is also an option
        align_precision=2, # Spatial alignment precision (0-4)
        unscaled=True,     # Prefer unscaled spatial match
        verbose=False
    )
    print(f"Video composed successfully: {output_video}")
except Exception as e:
    print(f"An error occurred during video composition: {e}")

```
The `composite_videos` function accepts most of the same parameters as the CLI `comp` command (e.g., `drift_interval`, `window`, `smooth`, `x_shift`, `y_shift`, `zero_shift`). Refer to its docstring or the CLI help for more details.

#### Thumbnail Detection

To find a thumbnail image within a larger image or video, use the `ThumbnailFinder` class.

```python
from vidkompy.align.core import ThumbnailFinder
from vidkompy.utils.logging import make_logger # Optional: for logging setup
from pathlib import Path

# Optional: Configure logging if you want to see detailed output like the CLI
# make_logger(name="my_thumbnail_search", verbose=True)

# Define paths
foreground_image = Path("path/to/your/thumbnail.jpg")
background_source = Path("path/to/your/background_video.mp4") # Can be an image or video

# Initialize the finder
finder = ThumbnailFinder()

try:
    result = finder.find_thumbnail(
        fg=str(foreground_image),
        bg=str(background_source),
        precision=2,       # Detection precision (0-4)
        num_frames=7,      # Max frames if background is a video
        unscaled=True,     # Prefer 100% scale matches
        verbose=False      # Set to True to see detailed console output during processing
    )

    # The 'result' object is a ThumbnailResult instance containing detection details
    if result and result.confidence > 0.5: # Example confidence check
        print(f"Thumbnail found with confidence: {result.confidence:.2f}")
        print(f"  Scale: {result.scale_fg_to_thumb:.2f}%")
        print(f"  Position (x, y): ({result.x_thumb_in_bg}, {result.y_thumb_in_bg})")
        # Access more details from the result object as needed
        # e.g., result.fg_size, result.bg_size, result.analysis_data
    else:
        print("Thumbnail not found or confidence too low.")

except Exception as e:
    print(f"An error occurred during thumbnail detection: {e}")

```
The `ThumbnailFinder.find_thumbnail()` method will print analysis to the console if `verbose=True` (or if a logger is configured and its verbosity implies it), similar to the CLI. The returned `ThumbnailResult` object (from `vidkompy.align.data_types`) contains detailed information about the match, including:
*   `confidence`: The confidence score of the match (0.0 to 1.0).
*   `scale_fg_to_thumb`: Percentage to scale foreground to match background.
*   `x_thumb_in_bg`, `y_thumb_in_bg`: Top-left coordinates of the thumbnail in the background.
*   `fg_size`, `bg_size`, `thumbnail_size`: Dimensions.
*   And more, including detailed `analysis_data`.

---
## Technical Deep-Dive

This section provides a more detailed look into `vidkompy`'s internal workings, architecture, algorithms, and contribution guidelines.

### How `vidkompy` Works: The Processing Pipeline

`vidkompy` processes videos in a multi-stage pipeline designed for precision and accuracy:

1.  **Video Analysis (Probing):**
    *   Both background (BG) and foreground (FG) videos are first probed using `ffprobe`.
    *   This extracts essential metadata: resolution, frames per second (FPS), duration, frame count, and audio stream information. This data informs subsequent processing steps.

2.  **Spatial Alignment (Where to place FG on BG):**
    *   To determine the (x, y) offset for the foreground video, `vidkompy` utilizes its advanced `align` module.
    *   Typically, sample frames (e.g., from the middle of the videos) are used.
    *   The `align` module employs a suite of algorithms (template matching, feature detection, etc.) with configurable precision to find the best spatial match of the foreground within the background. See "Thumbnail Detection Algorithms" below for details.
    *   This step yields the precise coordinates and any necessary scaling factor for the overlay. For video composition, it defaults to a balanced precision and prefers unscaled (100%) matches unless specified otherwise.

3.  **Temporal Alignment (How to map FG frames to BG frames):**
    *   This is crucial for synchronizing video content over time. `vidkompy` aims to map every foreground frame to a corresponding background frame.
    *   The primary approach involves:
        *   **Direct Frame Comparison:** The selected temporal alignment engine (e.g., `FullEngine`) compares actual pixel data between foreground and background frames.
        *   **Sliding Window & Bidirectional Matching:** To find the optimal frame pairings, the engine searches within a constrained window, typically performing passes both forwards and backwards through the video timelines. This helps ensure a globally consistent and monotonic alignment, preventing drift.
    *   The result is an alignment map `(fg_frame_idx -> bg_frame_idx)` that dictates which background frame to use for each foreground frame.

4.  **Video Composition (Creating the Output):**
    *   With spatial and temporal alignments determined, `vidkompy` proceeds to create the output video.
    *   It reads both video streams sequentially (which is significantly faster than random access).
    *   For each foreground frame, it fetches the corresponding background frame based on the temporal alignment map.
    *   The foreground frame is then overlaid onto the selected background frame at the correct spatial (x,y) position, scaled if necessary.
    *   Options like smooth blending at frame edges can be applied here.

5.  **Audio Integration:**
    *   After the silent video is composed, the audio track is added.
    *   `vidkompy` typically prioritizes the foreground video's audio track. If unavailable, it falls back to the background's audio.
    *   The chosen audio track is processed (e.g., trimmed or offset) to ensure it's perfectly synchronized with the newly composed video content.

### Modular Architecture

`vidkompy` is designed with a modular architecture to separate concerns and promote extensibility:

*   **`src/vidkompy/align` (Thumbnail Detection & Spatial Alignment):**
    *   **Core Logic (`core.py`):** Contains `ThumbnailFinder` which orchestrates the detection.
    *   **Algorithms (`algorithms.py`):** Implementations of various matching algorithms (Template Matching, Feature Matching, etc.).
    *   **Precision System (`precision.py`):** Manages multi-level precision analysis.
    *   **Frame Extraction (`frame_extractor.py`):** Handles reading frames from images/videos.
    *   **Data Types (`data_types.py`):** Defines structures like `ThumbnailResult`.
    *   **Display (`display.py`):** Formats and shows results.
    *   This module is used by the `comp` module for spatial alignment and can also be used independently via the `align` CLI command or programmatically.

*   **`src/vidkompy/comp` (Video Composition):**
    *   **Main Orchestration (`vidkompy.py`):** Contains `composite_videos` function and `AlignmentEngine` class that manages the composition pipeline.
    *   **Video Processing (`video.py`):** `VideoProcessor` class handles interactions with FFmpeg for probing, frame reading, and final video/audio encoding.
    *   **Temporal Alignment (`temporal.py`, `dtw_aligner.py`):** Implements temporal synchronization logic, including engines like `FullEngine` and `MaskEngine` (previously referred to as `TunnelSyncer` variants). These use techniques like direct pixel comparison within sliding windows.
    *   **(Legacy/Support Components like `fingerprint.py`):** While perceptual hashing (`pHash`, `AverageHash`, etc.) was previously a more central part of temporal alignment, the current primary engines (`FullEngine`, `MaskEngine`) focus on direct pixel comparison for higher accuracy in the `comp` module. Fingerprinting techniques are more relevant to the `align` module's feature-based methods if used.

*   **`src/vidkompy/utils` (Utilities):**
    *   Contains helper modules for logging, image manipulation (e.g., grayscale conversion), Numba-optimized operations, and enums.

**Key Design Principles:**
*   **Separation of Concerns:** Each module has a clear responsibility.
*   **Unified Spatial Alignment:** The `align` module provides consistent spatial alignment capabilities for all parts of `vidkompy`.
*   **Algorithm Flexibility:** Multiple algorithms are available, especially in the `align` module, with mechanisms for fallbacks or selection based on precision.
*   **Performance and Accuracy:** Optimized for both speed (e.g., sequential processing, Numba) and precision (e.g., direct pixel comparison, multi-level analysis).

### Core Algorithms Explained

`vidkompy` employs sophisticated algorithms for its tasks:

#### Thumbnail Detection / Spatial Alignment Algorithms (`align` module)

This module finds a smaller image (foreground/thumbnail) within a larger image or video (background). It uses a combination of techniques, often selected based on the chosen `precision` level:

1.  **Template Matching (`TemplateMatchingAlgorithm`):**
    *   **Method:** Slides the foreground image across the background image and calculates a similarity score (Normalized Cross-Correlation, `TM_CCOEFF_NORMED`) at each position.
    *   **Multi-Scale:** Can test various scales of the foreground in parallel (using `ThreadPoolExecutor`).
    *   **Optimization:** Critical parts may be JIT-compiled with Numba.
    *   **Bias:** Can prefer unscaled (100%) matches if confidence scores are close.

2.  **Feature Matching (`FeatureMatchingAlgorithm`):**
    *   **Method:** Detects keypoints (distinctive features) in both images using detectors like AKAZE (default), ORB, or SIFT. It then matches these keypoints.
    *   **Robustness:** Uses techniques like ratio tests and RANSAC (Random Sample Consensus) to filter out bad matches and estimate geometric transformations (e.g., affine, homography).
    *   **Confidence:** Calculated based on the number and quality of inlier matches.

3.  **Phase Correlation (`PhaseCorrelationAlgorithm`):**
    *   **Method:** Operates in the frequency domain (using Fast Fourier Transform - FFT). It's particularly good at finding translational shifts.
    *   **Sub-Pixel Accuracy:** Can achieve high precision by upsampling the phase correlation surface.
    *   **Integration:** Often used to refine results from other methods or in hybrid approaches.

4.  **Histogram Correlation (`HistogramCorrelationAlgorithm`):**
    *   **Method:** Compares color/intensity histograms of image regions.
    *   **Speed:** Very fast, often used for initial ballpark estimations or at low precision levels.
    *   **Robustness:** Normalized histograms provide some robustness to lighting changes.

5.  **Hybrid Matching (`HybridMatchingAlgorithm`):**
    *   **Method:** Intelligently combines results from multiple algorithms (e.g., feature, template, phase correlation).
    *   **Strategy:** May use a cascaded approach (e.g., features first, then template matching to refine). Weights results based on confidence and reliability.

6.  **Sub-Pixel Refinement (`SubPixelRefinementAlgorithm`):**
    *   **Method:** Takes an initial estimate (e.g., from template matching) and searches in a small neighborhood at sub-pixel offsets to fine-tune the position for maximum correlation.

**Multi-Precision Analysis System (`PrecisionAnalyzer`):**
The `align` module offers 5 precision levels, trading off speed for accuracy:
*   **Level 0 (Ballpark):** ~1ms. Uses primarily Histogram Correlation.
*   **Level 1 (Coarse):** ~10ms. Faster Template Matching with wider steps.
*   **Level 2 (Balanced):** ~25ms. Default. Often a combination like Feature Matching + Template Matching.
*   **Level 3 (Fine):** ~50ms. More comprehensive Hybrid methods.
*   **Level 4 (Precise):** ~200ms. Includes Sub-Pixel Refinement for maximum accuracy.

The system often performs a progressive analysis, starting with faster methods and refining with more precise ones. An "unscaled preference" logic is also applied: if an exact 100% scale match has high confidence, it might be preferred over a scaled match even if the latter's confidence is marginally higher, which is useful for direct overlays.

#### Video Composition Algorithms (`comp` module)

1.  **Spatial Alignment (Leveraging the `align` module):**
    *   As described above, the `comp` module uses `ThumbnailFinder` from the `align` module to determine the (x,y) offset and scale for the foreground video.
    *   For composition, it typically defaults to `precision=2` (Balanced) and `unscaled=True` (preferring 100% scale matches). These can be overridden by CLI parameters (`--align_precision`, `--unscaled false`).

2.  **Temporal Alignment Engines (`FullEngine`, `MaskEngine`):**
    These engines are responsible for creating the frame-to-frame mapping between the foreground and background videos.
    *   **Core Principle:** Direct pixel comparison within a sliding window to ensure monotonicity (preventing time travel or jitter) and thus achieving zero drift.
    *   **Bidirectional Matching:**
        *   A **Forward Pass** starts from the first foreground frame and finds the best match in the background within a search window, moving forward.
        *   A **Backward Pass** starts from the last foreground frame and searches backward.
        *   The results of these passes are often merged or used to create a robust alignment path.
    *   **Sliding Window Constraint:** The search for a match for the current foreground frame is limited to a "window" of background frames relative to the last match. This enforces temporal order and limits computational cost. The `window` parameter often controls this.
    *   **`FullEngine` (Default):**
        *   Compares the entire pixel content of the frames.
        *   Best for standard videos where the entire frame is relevant.
    *   **`MaskEngine`:**
        *   Similar to `FullEngine` but first generates a content mask for the foreground frame (e.g., identifying non-black areas).
        *   Comparison is then focused only on the pixels within this mask.
        *   Ideal for letterboxed or pillarboxed content, as it ignores black bars.

    *The parameter `drift_interval` is related to how often more significant realignments or checks might occur, though the core engines aim for continuous, drift-free alignment. The `window` parameter more directly controls the search space for frame-to-frame matching in these engines.*

### Performance Details

*   **Thumbnail Detection (`align` module):**
    *   Performance varies by precision level (from ~1ms for Level 0 to ~200ms for Level 4 per frame pair).
    *   Numba JIT compilation is used for critical functions.
    *   Parallel processing (ThreadPoolExecutor) is used for tasks like multi-scale template matching.

*   **Video Composition (`comp` module):**
    *   Typically processes video at speeds multiple times faster than real-time (e.g., 5-6x real-time for common HD videos with `FullEngine` or `MaskEngine`).
    *   **Sequential Frame Processing:** Reading frames sequentially for composition is 10-100x faster than random-access methods.
    *   **Zero Drift by Design:** The monotonic constraints of the temporal alignment engines (Full, Mask) inherently prevent temporal drift.
    *   **Direct Pixel Comparison:** Provides high accuracy for temporal alignment, though it can be computationally intensive (mitigated by sliding windows and optimizations).

### Development and Contribution

We welcome contributions to `vidkompy`! Here's how to get started:

#### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/twardoch/vidkompy.git
    cd vidkompy
    ```
2.  **Install Hatch:**
    If you don't have it, install Hatch, which is used for project and environment management:
    ```bash
    pip install hatch
    ```
3.  **Development Environment:**
    The project's dependencies and environments are defined in `pyproject.toml`. Hatch will manage these for you.

#### Key Development Commands

Run these commands from the root of the repository:

*   **Run Tests:**
    ```bash
    hatch run test
    ```
*   **Run Tests with Coverage Report:**
    ```bash
    hatch run test-cov
    ```
*   **Run Type Checking (MyPy):**
    ```bash
    hatch run type-check
    ```
*   **Check Formatting (Ruff) and Linting (Ruff):**
    ```bash
    hatch run lint
    ```
*   **Automatically Fix Formatting and Linting Issues (Ruff):**
    ```bash
    hatch run fix
    ```
    It's recommended to run `hatch run fix` before committing changes.

#### Coding Guidelines

Please adhere to the following guidelines when contributing:

*   **Style:**
    *   Follow **PEP 8** for code style and **PEP 257** for docstring conventions.
    *   Use clear, descriptive names for variables and functions.
    *   Write comprehensive docstrings for all public modules, classes, and functions, explaining *what* the code does and *why*. Include information on where and how it's used if relevant.
    *   Use **type hints** (e.g., `list`, `dict`, `str | None`) for function signatures and variables where appropriate.
    *   Employ f-strings for string formatting.
*   **Simplicity & Readability (PEP 20 - The Zen of Python):**
    *   Keep code simple, explicit, and readable. Prioritize clarity over overly clever or complex solutions.
    *   Favor flat structures over deeply nested ones.
*   **Modularity:** Encapsulate repeated logic into concise, single-purpose functions.
*   **`this_file` Convention:** For Python files, include a comment near the top indicating its path relative to the project root, e.g.:
    ```python
    # this_file: src/vidkompy/module/file.py
    ```
*   **Logging:** Use the `loguru` library for logging. Add meaningful log messages, especially for verbose/debug modes.
*   **Error Handling:** Handle potential errors gracefully. Validate inputs and assumptions.
*   **Dependencies:** Use `uv pip` for managing dependencies if working outside Hatch environments. All dependencies should be declared in `pyproject.toml`.
*   **Commits:** Write clear and concise commit messages.
*   **Updates to Documentation:** If your changes affect functionality or usage, please update `README.md` and relevant docstrings.
*   **Changelog:** Add an entry to `CHANGELOG.md` for significant changes.

Before making substantial changes or adding complex features, it's a good idea to open an issue to discuss your proposed approach.

---
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
