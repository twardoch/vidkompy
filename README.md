# `vidkompy`

**Intelligent Video Overlay and Synchronization**

`vidkompy` is a powerful command-line tool engineered to overlay a foreground video onto a background video with exceptional precision and automatic alignment. The system intelligently handles discrepancies in resolution, frame rate, duration, and audio, prioritizing content integrity and synchronization accuracy over raw processing speed.

The core philosophy of `vidkompy` is to treat the **foreground video as the definitive source of quality and timing**. All its frames are preserved without modification or re-timing. The background video is dynamically adapted—stretched, retimed, and selectively sampled—to synchronize perfectly with every frame of the foreground content, ensuring a seamless and coherent final output.

---

## Features

- **Automatic Spatial Alignment**: Intelligently detects the optimal x/y offset to position the foreground video within the background, even if they are cropped differently.
- **Advanced Temporal Synchronization**: Aligns videos with different start times, durations, and frame rates, eliminating temporal drift and ensuring content matches perfectly over time.
- **Foreground-First Principle**: Guarantees that every frame of the foreground video is included in the output, preserving its original timing and quality. The background video is adapted to match the foreground.
- **Drift-Free Alignment**: Utilizes Dynamic Time Warping (DTW) to create a globally optimal, monotonic alignment, preventing the common "drift-and-catchup" artifacts seen with simpler methods.
- **High-Performance Processing**: Leverages multi-core processing, perceptual hashing, and optimized video I/O to deliver results quickly.
- Frame fingerprinting is 100-1000x faster than traditional pixel-wise comparison.
- Sequential video composition is 10-100x faster than random-access methods.
- **Smart Audio Handling**: Automatically uses the foreground audio track if available, falling back to the background audio. The audio is correctly synchronized with the final video.
- **Flexible Operation Modes**: Supports specialized modes like `border` matching for aligning content based on visible background edges, and `smooth` blending for seamless visual integration.

## How It Works

The `vidkompy` pipeline is a multi-stage process designed for precision and accuracy:

1.  **Video Analysis**: The tool begins by probing both background (BG) and foreground (FG) videos using `ffprobe` to extract essential metadata: resolution, frames per second (FPS), duration, frame count, and audio stream information.

2.  **Spatial Alignment**: To determine _where_ to place the foreground on the background, `vidkompy` extracts a sample frame from the middle of each video (where content is most likely to be stable). It then calculates the optimal (x, y) offset.

3.  **Temporal Alignment**: This is the core of `vidkompy`. To determine _when_ to start the overlay and how to map frames over time, the tool generates "fingerprints" of frames from both videos and uses Dynamic Time Warping (DTW) to find the best alignment path. This ensures every foreground frame is matched to the most suitable background frame.

4.  **Video Composition**: Once the spatial and temporal alignments are known, `vidkompy` composes the final video. It reads both video streams sequentially (for maximum performance) and, for each foreground frame, fetches the corresponding background frame as determined by the alignment map. The foreground is then overlaid at the correct spatial position.

5.  **Audio Integration**: After the silent video is composed, `vidkompy` adds the appropriate audio track (preferring the foreground's audio) with the correct offset to ensure it's perfectly synchronized with the video content.

## The Algorithms

`vidkompy` employs several sophisticated algorithms to achieve its high-precision results.

### Frame Fingerprinting (Perceptual Hashing)

**TLDR:** Instead of comparing the millions of pixels in a frame, `vidkompy` creates a tiny, unique "fingerprint" (a hash) for each frame. Comparing these small fingerprints is thousands of times faster and smart enough to ignore minor changes from video compression.

---

The `FrameFingerprinter` module is designed for ultra-fast and robust frame comparison. It uses perceptual hashing, which generates a compact representation of a frame's visual structure.

The process works as follows:

1.  **Standardization**: The input frame is resized to a small, standard size (e.g., 64x64 pixels) and converted to grayscale. This ensures consistency and focuses on structural information over color.
2.  **Multi-Algorithm Hashing**: To improve robustness, `vidkompy` computes several types of perceptual hashes for each frame, as different algorithms are sensitive to different visual features:
- `pHash` (Perceptual Hash): Analyzes the frequency domain (using DCT), making it robust to changes in brightness, contrast, and gamma correction.
- `AverageHash`: Computes a hash based on the average color of the frame.
- `ColorMomentHash`: Captures the color distribution statistics of the frame.
- `MarrHildrethHash`: Detects edges and shapes, making it sensitive to structural features.
3.  **Combined Fingerprint**: The results from these hashers, along with a color histogram, are combined into a single "fingerprint" dictionary for the frame.
4.  **Comparison**: To compare two frames, their fingerprints are compared. The similarity is calculated using a weighted average of the normalized Hamming distance between their hashes and the correlation between their histograms. The weights are tuned based on the reliability of each hash type for video content. This entire process is parallelized across multiple CPU cores for maximum speed.

### Spatial Alignment (Template Matching)

**TLDR:** To find the correct position for the foreground video, the tool takes a screenshot from the middle of it and searches for that exact image within a screenshot from the background video.

---

Spatial alignment determines the `(x, y)` coordinates at which to overlay the foreground frame onto the background. `vidkompy` uses a highly accurate and efficient method based on template matching.

1.  **Frame Selection**: A single frame is extracted from the temporal midpoint of both the foreground and background videos. This is done to get a representative frame, avoiding potential opening/closing titles or black frames.
2.  **Grayscale Conversion**: The frames are converted to grayscale. This speeds up the matching process by 3x and makes the alignment more robust to minor color variations between the videos.
3.  **Template Matching**: The core of the alignment is `cv2.matchTemplate` using the `TM_CCOEFF_NORMED` method. This function effectively "slides" the smaller foreground frame image across the larger background frame image and calculates a normalized cross-correlation score at each position.
4.  **Locating the Best Match**: The position with the highest correlation score (from `cv2.minMaxLoc`) is considered the best match. This location `(x_offset, y_offset)` represents the top-left corner where the foreground should be placed. The confidence of this match is the correlation score itself, which typically approaches `1.0` for a perfect match.
5.  **Scaling**: The system checks if the foreground video is larger than the background. If so, it is scaled down to fit, and the scale factor is recorded.

### Temporal Alignment (Dynamic Time Warping)

**TLDR:** `vidkompy` aligns the two videos by finding the optimal "path" through time that perfectly syncs the foreground to the background. It intelligently stretches or compresses the background's timeline to match the foreground frame-by-frame, preventing any synchronization drift.

---

Temporal alignment is the most critical and complex part of `vidkompy`. The goal is to create a mapping `FrameAlignment(fg_frame_idx, bg_frame_idx)` for every single foreground frame. This is achieved using **Dynamic Time Warping (DTW)**, a powerful algorithm from time series analysis.

The DTW process is as follows:

1.  **Frame Sampling & Fingerprinting**: The tool samples frames from both videos and computes their perceptual fingerprints using the `FrameFingerprinter`. More frames are sampled from the background video to provide more alignment options.
2.  **Cost Matrix Construction**: A cost matrix is built where `cost(i, j)` is the "distance" (i.e., `1.0 - similarity`) between the fingerprint of foreground frame `i` and background frame `j`. This matrix represents all possible frame pairings.
3.  **DTW Recursion**: The DTW algorithm finds the lowest-cost path through this matrix from the start `(0, 0)` to the end `(n, m)`. A key constraint is **monotonicity**: the path can only move forward in time for both videos, never backward. This is what prevents out-of-sync jumps.
4.  **Sakoe-Chiba Band Constraint**: To make the calculation feasible for long videos (reducing complexity from $$O(N^2)$$to$$O(N \times w)$$), the algorithm only computes costs within a "window" or band around the matrix diagonal. This prevents unrealistic warping, assuming the videos are reasonably aligned to begin with.
5.  **Optimal Path Backtracking**: Once the cost matrix is filled, the algorithm backtracks from the end to find the sequence of frame pairings that resulted in the lowest total cost. This sequence is the optimal alignment path.
6.  **Full Alignment Map Generation**: The DTW path provides a sparse set of perfect matches. To get an alignment for _every_ foreground frame, `vidkompy` linearly interpolates the background frame indices between these key matches. This results in a smooth and continuous mapping, ensuring the background video playback speed adjusts dynamically to stay locked with the foreground.

This DTW-based approach guarantees a globally optimal and monotonic alignment, robustly handling speed variations, dropped frames, and other temporal distortions.

## Usage

### Prerequisites

You must have the **FFmpeg** binary installed on your system and accessible in your system's `PATH`. `vidkompy` depends on it for all video and audio processing tasks.

### Installation

The tool is a Python package. It is recommended to install it from the repository to get the latest version.

```bash
# Clone the repository
git clone https://github.com/twardoch/vidkompy.git
cd vidkompy

# Install using uv (or pip)
uv pip install .
```

### Command-Line Interface (CLI)

The tool is run from the command line, providing paths to the background and foreground videos.

**Basic Example:**

```bash
python -m vidkompy --bg /path/to/background.mp4 --fg /path/to/foreground.mp4
```

**CLI Help:**

```
$ python -m vidkompy -- --help
INFO: Showing help with the command 'vidkompy.py -- --help'.

NAME
vidkompy.py - Overlay foreground video onto background video with intelligent alignment.

SYNOPSIS
vidkompy.py BG FG <flags>

DESCRIPTION
Overlay foreground video onto background video with intelligent alignment.

POSITIONAL ARGUMENTS
BG
    Type: str | pathlib.Path
    Background video path
FG
    Type: str | pathlib.Path
    Foreground video path

FLAGS
-o, --output=OUTPUT
    Type: Optional[str | pathlib...
    Default: None
    Output video path (auto-generated if not provided)
-m, --margin=MARGIN
    Type: int
    Default: 8
    Border thickness for border matching mode (default: 8)
-s, --smooth=SMOOTH
    Type: bool
    Default: False
    Enable smooth blending at frame edges
-g, --gpu=GPU
    Type: bool
    Default: False
    Enable GPU acceleration (future feature)
-v, --verbose=VERBOSE
    Type: bool
    Default: False
    Enable verbose logging

NOTES
You can also use flags syntax for POSITIONAL ARGUMENTS
```

## Performance

Recent updates have significantly improved `vidkompy`'s performance and accuracy:

- **Drift Elimination**: An adaptive keyframe density calculation based on FPS differences prevents temporal drift, and the default keyframe target has been lowered from 2000 to 200 for a better speed/accuracy balance.
- **Optimized Compositing**: The video writing process was re-engineered to use sequential frame reading instead of random access, yielding a **10-100x speedup** in the final composition stage.
- **Parallel Processing**: The most computationally expensive tasks, like building the similarity cost matrix, are parallelized to leverage all available CPU cores.
- **Benchmarking**: The project includes a `benchmark.py` script to test various configurations and measure performance metrics, ensuring continuous improvement.

## Development

To contribute to `vidkompy`, set up a development environment using `hatch`.

### Setup

1.  Clone the repository.
2.  Ensure you have `hatch` installed (`pip install hatch`).
3.  The project is managed through `hatch` environments defined in `pyproject.toml`.

### Key Commands

Run these commands from the root of the repository.

- **Run Tests**:

```bash
hatch run test
```

- **Run Tests with Coverage Report**:

```bash
hatch run test-cov
```

- **Run Type Checking**:

```bash
hatch run type-check
```

- **Check Formatting and Linting**:

```bash
hatch run lint
```

- **Automatically Fix Formatting and Linting Issues**:

```bash
hatch run fix
```

## License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
