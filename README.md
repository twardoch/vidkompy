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

`vidkompy` offers five temporal alignment engines, each with different trade-offs between speed, accuracy, and approach:
- **Fast** (default): Quick processing with perceptual hashing
- **Precise**: Maximum accuracy with multi-resolution alignment
- **Mask**: Enhanced precise engine with explicit masking
- **Tunnel Full**: Direct pixel comparison with sliding windows
- **Tunnel Mask**: Pixel comparison focused on content regions

---

Temporal alignment is the most critical and complex part of `vidkompy`. The goal is to create a mapping `FrameAlignment(fg_frame_idx, bg_frame_idx)` for every single foreground frame. `vidkompy` provides five distinct engines for this task:

#### Fast Engine (Default)

The **Fast Engine** uses **Dynamic Time Warping (DTW)** with perceptual hashing for efficient alignment:

1.  **Frame Sampling & Fingerprinting**: The tool samples frames sparsely based on the `max_keyframes` parameter and computes their perceptual fingerprints using multiple hash algorithms (pHash, AverageHash, ColorMomentHash, MarrHildrethHash).
2.  **Cost Matrix Construction**: A cost matrix is built where `cost(i, j)` is the "distance" (i.e., `1.0 - similarity`) between the fingerprint of foreground frame `i` and background frame `j`.
3.  **DTW with Constraints**: The DTW algorithm finds the lowest-cost path through this matrix with:
   - **Monotonicity**: The path can only move forward in time, preventing temporal jumps
   - **Sakoe-Chiba Band**: Constrains the search to a window around the diagonal (reduces complexity from O(N²) to O(N×w))
4.  **Direct Mapping Mode**: With `max_keyframes=1` (default in fast mode), the engine forces direct frame mapping to eliminate drift entirely.
5.  **Interpolation**: For sparse sampling, the engine linearly interpolates between matched keyframes to create a complete alignment map.

**Characteristics:**
- Processing time: ~15 seconds for an 8-second video
- Minimal drift with direct mapping mode
- Suitable for most use cases

#### Precise Engine (Advanced)

The **Precise Engine** implements a sophisticated multi-resolution approach for maximum accuracy. **Recent enhancements include improved drift correction using polynomial models, adaptive blending, and Savitzky-Golay smoothing to address temporal inconsistencies.**

1.  **Multi-Resolution Hierarchical Alignment**:
   - Creates temporal pyramids at multiple resolutions (1/16, 1/8, 1/4, 1/2, full)
   - Performs coarse-to-fine alignment, starting at the lowest resolution
   - Each level refines the previous level's mapping
   - Applies drift correction (now enhanced) every 100 frames

2.  **Keyframe Detection and Anchoring**:
   - Automatically detects keyframes based on temporal changes using Gaussian filtering
   - Aligns keyframes between videos as anchor points
   - Forces alignment at keyframes to prevent long-range drift
   - Detects scene changes and content transitions

3.  **Bidirectional DTW**:
   - Runs DTW in both forward and backward directions
   - Averages the two alignment paths to reduce systematic bias
   - Provides more robust alignment for videos with varying content

4.  **Sliding Window Refinement**:
   - Refines alignment in 30-frame windows
   - Searches locally for optimal alignment adjustments
   - Applies Gaussian smoothing for smooth transitions
   - Ensures strict monotonicity throughout

5.  **Confidence-Based Weighting**:
   - Computes confidence scores for each alignment
   - Weights multiple alignment methods based on their confidence
   - Combines results for optimal accuracy

**Characteristics:**
- Processing time: ~5 minutes for an 8-second video (includes full frame extraction)
- Virtually eliminates all temporal drift
- Handles complex scenarios with varying frame rates and content changes
- Best for critical applications requiring perfect synchronization

#### Tunnel Full Engine (Direct Comparison)

The **Tunnel Full Engine** uses direct pixel-by-pixel frame comparison with a sliding window approach:

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
- Processing time: Varies with window size and video resolution
- Zero drift by design due to monotonic constraints
- Best for videos with minimal compression artifacts
- Suitable when perceptual hashing misses subtle details

#### Tunnel Mask Engine (Content-Focused Comparison)

The **Tunnel Mask Engine** extends the tunnel approach with intelligent masking:

1. **Content Mask Generation**:
   - Automatically detects content regions (non-black areas) in FG frames
   - Creates binary mask to focus comparison on actual content
   - Helps with letterboxed or pillarboxed videos

2. **Masked Comparison**:
   - Only compares pixels within the mask region
   - Ignores black borders and letterboxing
   - More robust for videos with varying aspect ratios

3. **Same Bidirectional Approach**:
   - Uses forward and backward passes like Tunnel Full
   - Applies mask during all comparisons
   - Maintains monotonicity constraints

**Characteristics:**
- Similar performance to Tunnel Full
- Better handling of videos with black borders
- More accurate for content with letterboxing
- Ideal for videos where content doesn't fill the entire frame

#### Engine Comparison

| Aspect | Fast | Precise | Mask | Tunnel Full | Tunnel Mask |
|--------|------|---------|------|-------------|-------------|
| **Algorithm** | DTW + hashing | Multi-res DTW | Multi-res + mask | Direct pixel | Masked pixel |
| **Speed** | ~2x real-time | ~40x real-time | ~40x real-time | ~10-20x real-time | ~10-20x real-time |
| **Drift** | Minimal | Minimal | Minimal | Zero (monotonic) | Zero (monotonic) |
| **Memory** | Low | High | High | Medium | Medium |
| **Best For** | Quick results | Complex videos | Cropped content | Clean sources | Letterboxed |

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
# Fast engine with direct mapping (default, no drift)
python -m vidkompy --bg background.mp4 --fg foreground.mp4

# Precise engine for maximum accuracy (slower but perfect sync)
python -m vidkompy --bg background.mp4 --fg foreground.mp4 --engine precise

# Tunnel engines for direct pixel comparison (no drift)
python -m vidkompy --bg bg.mp4 --fg fg.mp4 --engine tunnel_full --window 60
python -m vidkompy --bg bg.mp4 --fg letterboxed.mp4 --engine tunnel_mask

# Custom output path
python -m vidkompy --bg bg.mp4 --fg fg.mp4 --output result.mp4
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

| Engine | Processing Time | Speed Ratio | Drift at 1s | Drift at End | Notes |
|--------|----------------|-------------|-------------|--------------|-------|
| **Fast (default)** | 15.8 seconds | ~2x real-time | Minimal | Minimal | Direct mapping prevents drift |
| **Precise** | 5m 18s | ~40x real-time | Less drift | Minimal | Full frame extraction + multi-resolution |

**Key Performance Insights:**

- **Fast Engine**: Processes at approximately 2x real-time speed. With `max_keyframes=1` (default), it uses direct frame mapping which completely eliminates drift while maintaining fast performance.

- **Precise Engine**: While significantly slower (~40x real-time), it provides superior alignment accuracy, especially for complex videos. Interestingly, it shows less drift at the 1-second mark compared to the fast engine, though both engines perform well at video endpoints.

### Technical Optimizations

- **Drift Elimination**: The fast engine now defaults to `max_keyframes=1`, forcing direct frame-to-frame mapping that eliminates temporal drift entirely.
- **Optimized Compositing**: Sequential frame reading instead of random access yields a **10-100x speedup** in the final composition stage.
- **Parallel Processing**: Frame fingerprinting and cost matrix computation leverage all available CPU cores.
- **Perceptual Hashing**: Frame comparison is **100-1000x faster** than pixel-wise methods while maintaining accuracy.
- **Memory Efficiency**: The fast engine uses streaming processing, while the precise engine trades memory for accuracy by loading all frames.

### Choosing the Right Engine

**Use the Fast Engine (default) when:**
- You need quick results (2x real-time processing)
- The videos are already reasonably synchronized
- Minor imperfections are acceptable
- Processing many videos in batch

**Use the Precise Engine when:**
- Perfect synchronization is critical
- Videos have complex timing variations
- Content quality justifies longer processing time
- Working with professionally edited content

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
