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

Instead of comparing the millions of pixels in a frame, `vidkompy` creates a tiny, unique "fingerprint" (a hash) for each frame. Comparing these small fingerprints is thousands of times faster and smart enough to ignore minor changes from video compression.

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

To find the correct position for the foreground video, the tool takes a screenshot from the middle of it and searches for that exact image within a screenshot from the background video.

---

Spatial alignment determines the `(x, y)` coordinates at which to overlay the foreground frame onto the background. `vidkompy` uses a highly accurate and efficient method based on template matching.

1.  **Frame Selection**: A single frame is extracted from the temporal midpoint of both the foreground and background videos. This is done to get a representative frame, avoiding potential opening/closing titles or black frames.
2.  **Grayscale Conversion**: The frames are converted to grayscale. This speeds up the matching process by 3x and makes the alignment more robust to minor color variations between the videos.
3.  **Template Matching**: The core of the alignment is `cv2.matchTemplate` using the `TM_CCOEFF_NORMED` method. This function effectively "slides" the smaller foreground frame image across the larger background frame image and calculates a normalized cross-correlation score at each position.
4.  **Locating the Best Match**: The position with the highest correlation score (from `cv2.minMaxLoc`) is considered the best match. This location `(x_offset, y_offset)` represents the top-left corner where the foreground should be placed. The confidence of this match is the correlation score itself, which typically approaches `1.0` for a perfect match.
5.  **Scaling**: The system checks if the foreground video is larger than the background. If so, it is scaled down to fit, and the scale factor is recorded.

### Temporal Alignment Engines

`vidkompy` offers two high-performance temporal alignment engines optimized for different scenarios:
- **Full** (default): Direct pixel comparison with sliding windows for maximum accuracy
- **Mask**: Content-focused comparison with intelligent masking for letterboxed content

---

Temporal alignment is the most critical and complex part of `vidkompy`. The goal is to create a mapping `FrameAlignment(fg_frame_idx, bg_frame_idx)` for every single foreground frame. `vidkompy` provides two optimized engines for this task:

#### Full Engine (Default)

The **Full Engine** uses direct pixel-by-pixel frame comparison with a sliding window approach for maximum accuracy:

1. **Bidirectional Matching**:
   - **Forward Pass**: Starts from the first FG frame, searches for best match in BG within a sliding window
   - **Backward Pass**: Starts from the last FG frame, searches backward
   - Merges both passes for robust alignment

2. **Sliding Window Constraint**:
   - Enforces monotonicity by design - can only search forward from the last matched frame
   - Window size controls the maximum temporal displacement
   - Prevents temporal jumps and ensures smooth progression

3. **Direct Pixel Comparison**:
   - Compares actual pixel values between FG and BG frames
   - No information loss from hashing or fingerprinting
   - More sensitive to compression artifacts but potentially more accurate

**Characteristics:**
- Processing time: ~40 seconds for an 8-second video (d10-w10 configuration)
- Zero drift by design due to monotonic constraints
- Perfect confidence scores (1.000)
- Best overall performance for standard videos

#### Mask Engine (Content-Focused)

The **Mask Engine** extends the Full engine approach with intelligent masking for letterboxed or pillarboxed content:

1. **Content Mask Generation**:
   - Automatically detects content regions (non-black areas) in FG frames
   - Creates binary mask to focus comparison on actual content
   - Helps with letterboxed or pillarboxed videos

2. **Masked Comparison**:
   - Only compares pixels within the mask region
   - Ignores black borders and letterboxing
   - More robust for videos with varying aspect ratios

3. **Same Bidirectional Approach**:
   - Uses forward and backward passes like Full engine
   - Applies mask during all comparisons
   - Maintains monotonicity constraints

**Characteristics:**
- Processing time: ~45 seconds for an 8-second video (d10-w10 configuration) 
- Perfect confidence scores (1.000)
- Better handling of videos with black borders
- Ideal for videos where content doesn't fill the entire frame


#### Engine Comparison

| Aspect | Full | Mask |
|--------|------|------|
| **Algorithm** | Direct pixel comparison | Masked pixel comparison |
| **Speed** | ~5x real-time | ~5x real-time |
| **Drift** | Zero (monotonic) | Zero (monotonic) |
| **Memory** | Medium | Medium |
| **Confidence** | Perfect (1.000) | Perfect (1.000) |
| **Best For** | Standard videos | Letterboxed/pillarboxed content |

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

**Basic Examples:**

```bash
# Full engine (default) - direct pixel comparison with zero drift
python -m vidkompy --bg background.mp4 --fg foreground.mp4

# Mask engine for letterboxed/pillarboxed content
python -m vidkompy --bg background.mp4 --fg foreground.mp4 --engine mask

# Custom output path
python -m vidkompy --bg bg.mp4 --fg fg.mp4 --output result.mp4

# Fine-tune performance with drift interval and window size
python -m vidkompy --bg bg.mp4 --fg fg.mp4 --drift_interval 10 --window 10
```

**CLI Help:**

```
INFO: Showing help with the command '__main__.py -- --help'.

NAME
    __main__.py - Overlay foreground video onto background video with intelligent alignment.

SYNOPSIS
    __main__.py BG FG <flags>

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
    -e, --engine=ENGINE
        Type: str
        Default: 'fast'
        Temporal alignment engine - 'fast', 'precise', 'mask', 'tunnel_full', or 'tunnel_mask' (default: 'fast')
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

### Real-World Performance Comparison

Based on actual benchmarks with an 8-second test video (1920x1080 background, 1920x870 foreground, ~480 frames):

| Engine | Processing Time | Speed Ratio | Confidence | Notes |
|--------|----------------|-------------|------------|-------|
| **Full (default)** | 40.9 seconds | ~5x real-time | 1.000 (perfect) | Fastest overall with zero drift |
| **Mask** | 45.8 seconds | ~6x real-time | 1.000 (perfect) | Best for letterboxed content |

**Key Performance Insights:**

- **Full Engine**: Delivers perfect confidence scores (1.000) with ~5x real-time processing. Uses direct frame mapping which completely eliminates drift while maintaining excellent performance.

- **Mask Engine**: Slightly slower than Full engine but achieves perfect confidence. Ideal for content with black borders or letterboxing where content-focused comparison is beneficial.

### Technical Optimizations

- **Zero Drift Design**: Both engines use sliding window constraints that enforce monotonicity by design, completely eliminating temporal drift.
- **Optimized Compositing**: Sequential frame reading instead of random access yields a **10-100x speedup** in the final composition stage.
- **Direct Pixel Comparison**: Frame comparison uses actual pixel values without information loss from hashing or compression.
- **Bidirectional Matching**: Forward and backward passes are merged for robust alignment results.
- **Efficient Memory Usage**: Both engines use streaming processing with reasonable memory footprints.

### Choosing the Right Engine

**Use the Full Engine (default) when:**
- Working with standard videos without letterboxing
- You need the fastest processing with perfect accuracy
- Videos have consistent content filling the frame
- General-purpose video synchronization

**Use the Mask Engine when:**
- Working with letterboxed or pillarboxed content
- Videos have significant black borders
- Content doesn't fill the entire frame
- Aspect ratio mismatches between foreground and background

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
