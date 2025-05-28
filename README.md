# `vidkompy`

[![PyPI](https://img.shields.io/pypi/v/vidkompy.svg)](https://pypi.org/project/vidkompy/) [![License](https://img.shields.io/github/license/twardoch/vidkompy.svg)](

**Intelligent Video Overlay and Synchronization**

`vidkompy` is a powerful command-line tool engineered to overlay a foreground video onto a background video with exceptional precision and automatic alignment. The system intelligently handles discrepancies in resolution, frame rate, duration, and audio, prioritizing content integrity and synchronization accuracy over raw processing speed.

The core philosophy of `vidkompy` is to treat the **foreground video as the definitive source of quality and timing**. All its frames are preserved without modification or re-timing. The background video is dynamically adapted—stretched, retimed, and selectively sampled—to synchronize perfectly with every frame of the foreground content, ensuring a seamless and coherent final output.

---

## 1. Features

### 1.1. Video Composition

- **Automatic Spatial Alignment**: Intelligently detects the optimal x/y offset to position the foreground video within the background, even if they are cropped differently.
- **Advanced Temporal Synchronization**: Aligns videos with different start times, durations, and frame rates, eliminating temporal drift and ensuring content matches perfectly over time.
- **Foreground-First Principle**: Guarantees that every frame of the foreground video is included in the output, preserving its original timing and quality. The background video is adapted to match the foreground.
- **Drift-Free Alignment**: Uses optimized sliding window algorithms to create globally optimal, monotonic alignment, preventing the common "drift-and-catchup" artifacts.
- **High-Performance Processing**: Leverages multi-core processing, direct pixel comparison, and optimized video I/O to deliver results quickly.
- Sequential video composition is 10-100x faster than random-access methods.
- **Smart Audio Handling**: Automatically uses the foreground audio track if available, falling back to the background audio. The audio is correctly synchronized with the final video.
- **Flexible Operation Modes**: Supports specialized modes like `mask` for letterboxed content and `smooth` blending for seamless visual integration.

### 1.2. Thumbnail Detection (`align` module)

- **Multi-Precision Analysis**: Progressive refinement system with 5 precision levels (1ms to 200ms processing time)
- **Advanced Algorithm Suite**: 6 specialized algorithms including template matching, feature matching, phase correlation, and hybrid approaches
- **Parallel Processing**: Numba JIT compilation and ThreadPoolExecutor optimization for maximum performance
- **Multiple Detection Methods**: Template matching, AKAZE/ORB/SIFT feature detection, histogram correlation, and sub-pixel refinement
- **Intelligent Fallbacks**: Automatic algorithm selection and graceful fallback between detection methods
- **Rich Analysis**: Comprehensive confidence metrics, processing statistics, and alternative result comparison

## 2. How It Works

The `vidkompy` pipeline is a multi-stage process designed for precision and accuracy:

1.  **Video Analysis**: The tool begins by probing both background (BG) and foreground (FG) videos using `ffprobe` to extract essential metadata: resolution, frames per second (FPS), duration, frame count, and audio stream information.

2.  **Spatial Alignment**: To determine _where_ to place the foreground on the background, `vidkompy` extracts a sample frame from the middle of each video (where content is most likely to be stable). It then calculates the optimal (x, y) offset.

3.  **Temporal Alignment**: This is the core of `vidkompy`. To determine _when_ to start the overlay and how to map frames over time, the tool generates "fingerprints" of frames from both videos and uses Dynamic Time Warping (DTW) to find the best alignment path. This ensures every foreground frame is matched to the most suitable background frame.

4.  **Video Composition**: Once the spatial and temporal alignments are known, `vidkompy` composes the final video. It reads both video streams sequentially (for maximum performance) and, for each foreground frame, fetches the corresponding background frame as determined by the alignment map. The foreground is then overlaid at the correct spatial position.

5.  **Audio Integration**: After the silent video is composed, `vidkompy` adds the appropriate audio track (preferring the foreground's audio) with the correct offset to ensure it's perfectly synchronized with the video content.

## 3. Modular Architecture

`vidkompy` features a clean, modular architecture that separates concerns and enables easy extension:

### 3.1. Thumbnail Detection (`align` module)

- **Core Classes**: `ThumbnailFinder` orchestrates the detection process
- **Algorithms**: Multiple detection algorithms including template matching, feature matching, phase correlation, and hybrid approaches
- **Precision System**: Progressive refinement with 5 precision levels (0=ballpark ~1ms to 4=precise ~200ms)
- **Performance**: Numba-optimized computations with parallel processing support
- **Rich Output**: Comprehensive result display with confidence metrics and processing statistics

### 3.2. Video Composition (`comp` module)

- **Alignment Engines**: Specialized engines for different content types (Full, Mask)
- **Unified Spatial Alignment**: Uses the advanced `align` module for consistent, high-quality spatial detection
- **Temporal Synchronization**: Advanced DTW-based frame mapping with zero-drift constraints
- **Audio Processing**: Intelligent audio track selection and synchronization
- **Performance Optimization**: Sequential processing with 10-100x speedup over random access

### 3.3. Key Design Principles

- **Unified Spatial Alignment**: Single spatial alignment implementation across all modules using the `align` module
- **Separation of Concerns**: Each module handles a specific aspect of the processing pipeline
- **Algorithm Flexibility**: Multiple algorithms available with automatic fallbacks
- **Performance Focus**: Optimized for both speed and accuracy with configurable trade-offs
- **Extensibility**: Clean interfaces allow easy addition of new algorithms and features

## 4. The Algorithms

`vidkompy` employs several sophisticated algorithms to achieve its high-precision results.

### 4.1. Thumbnail Detection Algorithms (`align` module)

The modular thumbnail detection system provides multiple specialized algorithms for robust image matching:

#### 4.1.1. Template Matching Algorithm

- **Multi-Scale Processing**: Tests multiple scale factors in parallel using ThreadPoolExecutor
- **Ballpark Estimation**: Ultra-fast histogram correlation for initial scale estimation (~1ms)
- **Parallel Optimization**: Numba JIT compilation for critical computational functions
- **unscaled Bias**: Small preference for exact scale matches when confidence is similar
- **Normalized Cross-Correlation**: Uses OpenCV's TM_CCOEFF_NORMED for reliable matching

#### 4.1.2. Feature Matching Algorithm

- **Multiple Detectors**: Supports AKAZE (default), ORB, and SIFT feature detectors
- **Robust Matching**: Ratio test filtering and RANSAC-based outlier rejection
- **Transformation Estimation**: Uses estimateAffinePartial2D and homography methods
- **Confidence Calculation**: Based on inlier ratios and geometric consistency
- **Automatic Fallbacks**: Graceful degradation when detection methods fail

#### 4.1.3. Phase Correlation Algorithm

- **FFT-Based Processing**: Uses scikit-image's phase_cross_correlation
- **Sub-Pixel Accuracy**: 10x upsampling factor for precise position detection
- **Scale Integration**: Works with scale estimates from other algorithms
- **Error Conversion**: Transforms phase correlation error into confidence metrics

#### 4.1.4. Hybrid Matching Algorithm

- **Multi-Method Combination**: Intelligently combines feature, template, and phase correlation
- **Weighted Selection**: Results weighted by confidence and method reliability
- **Adaptive Strategy**: Adjusts approach based on initial feature detection success
- **Cascaded Processing**: Feature → Template → Phase correlation pipeline

#### 4.1.5. Histogram Correlation Algorithm

- **Ultra-Fast Processing**: Provides ballpark scale estimation in ~1ms
- **Multi-Region Sampling**: Tests correlation across multiple image regions
- **Normalized Histograms**: Robust to brightness and contrast variations
- **Numba Optimization**: JIT-compiled correlation functions for maximum speed

#### 4.1.6. Sub-Pixel Refinement Algorithm

- **Precision Enhancement**: Refines position estimates with sub-pixel accuracy
- **Local Search**: Tests sub-pixel offsets around initial estimates
- **Normalized Correlation**: Direct correlation calculation for fine-tuning
- **Quality Improvement**: Enhances results from other algorithms

### 4.2. Multi-Precision Analysis System

The system offers 5 precision levels with different speed/accuracy trade-offs:

- **Level 0 (Ballpark)**: Histogram correlation only (~1ms)
- **Level 1 (Coarse)**: Parallel template matching with wide steps (~10ms)
- **Level 2 (Balanced)**: Feature + template matching combination (~25ms, default)
- **Level 3 (Fine)**: Hybrid algorithm with multiple methods (~50ms)
- **Level 4 (Precise)**: Sub-pixel refinement for maximum accuracy (~200ms)

### 4.3. Video Composition Algorithms (`comp` module)

#### 4.3.1. Frame Fingerprinting (Perceptual Hashing)

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

### 4.4. Unified Spatial Alignment System

`vidkompy` now uses a unified spatial alignment system across both video composition and standalone thumbnail detection, leveraging the advanced multi-algorithm approach from the `align` module.

---

**Spatial alignment determines the `(x, y)` coordinates at which to overlay the foreground frame onto the background. The system now uses the advanced `ThumbnailFinder` from the `align` module for both video composition and standalone analysis.**

#### 4.4.1. Multi-Algorithm Approach

The unified system provides 6 specialized algorithms with automatic fallbacks:

1. **Template Matching**: Normalized cross-correlation with multi-scale parallel processing
2. **Feature Matching**: AKAZE/ORB/SIFT feature detection with geometric validation
3. **Phase Correlation**: FFT-based sub-pixel accuracy with 10x upsampling
4. **Histogram Correlation**: Ultra-fast ballpark estimation (~1ms)
5. **Hybrid Matching**: Intelligent combination of multiple methods
6. **Sub-Pixel Refinement**: Maximum precision enhancement

#### 4.4.2. Multi-Precision System

The system offers 5 precision levels for optimal speed/accuracy trade-offs:

- **Level 0 (Ballpark)**: Histogram correlation (~1ms) - ultra-fast scale estimation
- **Level 1 (Coarse)**: Parallel template matching (~10ms) - quick detection
- **Level 2 (Balanced)**: Feature + template combination (~25ms) - **default for video composition**
- **Level 3 (Fine)**: Hybrid multi-algorithm (~50ms) - high quality detection
- **Level 4 (Precise)**: Sub-pixel refinement (~200ms) - maximum accuracy

#### 4.4.3. unscaled Preference

The system includes intelligent **unscaled preference** that slightly favors 100% scale matches when confidence scores are similar, ideal for video composition where exact-size overlays are common.

#### 4.4.4. Benefits of Unified System

- **Better Accuracy**: Multi-algorithm approach with automatic fallbacks
- **Multi-Scale Detection**: Finds thumbnails at various scales, not just 1:1
- **Robustness**: Multiple detection methods handle different video artifacts
- **Performance Options**: User-configurable precision levels
- **Consistency**: Same spatial alignment quality for both composition and analysis

### 4.5. Temporal Alignment Engines

`vidkompy` offers two high-performance temporal alignment engines optimized for different scenarios:

- **Full** (default): Direct pixel comparison with sliding windows for maximum accuracy
- **Mask**: Content-focused comparison with intelligent masking for letterboxed content

---

Temporal alignment is the most critical and complex part of `vidkompy`. The goal is to create a mapping `FrameAlignment(fg_frame_idx, bg_frame_idx)` for every single foreground frame. `vidkompy` provides two optimized engines for this task:

#### 4.5.1. Full Engine (Default)

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

#### 4.5.2. Mask Engine (Content-Focused)

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

#### 4.5.3. Engine Comparison

| Aspect         | Full                    | Mask                            |
| -------------- | ----------------------- | ------------------------------- |
| **Algorithm**  | Direct pixel comparison | Masked pixel comparison         |
| **Speed**      | ~5x real-time           | ~5x real-time                   |
| **Drift**      | Zero (monotonic)        | Zero (monotonic)                |
| **Memory**     | Medium                  | Medium                          |
| **Confidence** | Perfect (1.000)         | Perfect (1.000)                 |
| **Best For**   | Standard videos         | Letterboxed/pillarboxed content |

## 5. Detailed Process Example

This section provides a precise walkthrough of how `vidkompy` arrives at its thumbnail detection results, using an actual example analysis.

### 5.1. Example Analysis: fg1.mp4 → bg1.mp4

**Input Parameters:**
```bash
python -m vidkompy align --fg tests/fg1.mp4 --bg tests/bg1.mp4 -p 3 -n 12 --verbose
```

**Step 1: Frame Extraction**
- **Foreground**: Extracted 12 frames from fg1.mp4 (1920×684 resolution)
- **Background**: Extracted 12 frames from bg1.mp4 (1920×1080 resolution)
- **Processing**: Converted frames to grayscale for analysis

**Step 2: Multi-Level Precision Analysis**
The system performed progressive analysis with 4 precision levels on the first frame pair:

- **Level 0 (Ballpark ~1ms)**: Histogram correlation → scale ≈ 70.0%, confidence = 0.951
- **Level 1 (Coarse ~10ms)**: Parallel template matching → scale = 84.00%, pos = (140, 173), confidence = 0.567  
- **Level 2 (Balanced ~25ms)**: Feature + template matching → scale = 100.51%, pos = (-3, 104), confidence = 0.875
- **Level 3 (Fine ~50ms)**: Hybrid multi-algorithm → scale = 98.97%, pos = (9, 114), confidence = 0.970

**Step 3: unscaled Verification**
Since `unscaled=True` (default), the system performed an additional check for exact 100% scale matching:
- **unscaled Test**: Template matching at scale = 1.0 (100%)
- **Result**: confidence = 0.981, position = (0, 109)
- **Decision**: unscaled result (98.11%) was chosen over best precision result (97.0%)

**Step 4: Final Result Calculation**
The algorithm determined the optimal alignment:

- **Confidence**: 98.11% (from unscaled matching)
- **Scale**: 100% (no scaling needed)
- **Position**: Foreground should be placed at (0, 109) in background coordinates
- **Interpretation**: The 1920×684 foreground fits perfectly within the 1920×1080 background, centered horizontally (x=0) and positioned 109 pixels down from the top (y=109)

**Step 5: Transformation Mathematics**
The system calculated bidirectional transformations:

*Forward (FG → BG):*
- Scale FG by 100% → 1920×684 thumbnail
- Place thumbnail at (0, 109) in BG coordinates

*Reverse (BG → FG):*
- Scale BG by 100% → 1920×1080 (no change)  
- FG appears at (0, -109) relative to scaled BG
- This means the BG extends 109 pixels above and 287 pixels below the FG content

**Why This Result Makes Sense:**
The analysis reveals that fg1.mp4 is a horizontally letterboxed version of bg1.mp4. The foreground (1920×684) represents the central content area of the background (1920×1080), with the background providing additional vertical space (396 pixels total: 109 above + 287 below). The 98.11% confidence indicates a near-perfect match with excellent spatial alignment.

## 6. Usage

### 6.1. Prerequisites

You must have the **FFmpeg** binary installed on your system and accessible in your system's `PATH`. `vidkompy` depends on it for all video and audio processing tasks.

### 6.2. Installation

The tool is a Python package. It is recommended to install it from the repository to get the latest version.

```bash
# Clone the repository
git clone https://github.com/twardoch/vidkompy.git
cd vidkompy

# Install using uv (or pip)
uv pip install .
```

### 6.3. Command-Line Interface (CLI)

`vidkompy` offers two main commands: video composition and thumbnail detection.

**Video Composition Examples:**

```bash
# Full engine (default) - direct pixel comparison with zero drift
python -m vidkompy comp --bg background.mp4 --fg foreground.mp4

# Mask engine for letterboxed/pillarboxed content
python -m vidkompy comp --bg background.mp4 --fg foreground.mp4 --engine mask

# Custom output path
python -m vidkompy comp --bg bg.mp4 --fg fg.mp4 --output result.mp4

# Fine-tune performance with drift interval and window size
python -m vidkompy comp --bg bg.mp4 --fg fg.mp4 --drift_interval 10 --window 10

# High-precision spatial alignment (slower but more accurate)
python -m vidkompy comp --bg bg.mp4 --fg fg.mp4 --align_precision 4

# Multi-scale spatial alignment (allows scale detection)
python -m vidkompy comp --bg bg.mp4 --fg fg.mp4 --unscaled false
```

**Thumbnail Detection Examples:**

The refactored thumbnail detection system now uses the `align` command with enhanced precision levels:

```bash
# Find thumbnail in image (new align command)
python -m vidkompy align foreground.jpg background.jpg

# Find thumbnail in video frame
python -m vidkompy align foreground.jpg background.mp4

# Precision levels 0-4 (0=ballpark ~1ms, 2=balanced ~25ms, 4=precise ~200ms)
python -m vidkompy align foreground.jpg background.jpg --precision 4

# Enable verbose logging for detailed analysis
python -m vidkompy align foreground.jpg background.jpg --verbose

# Multi-scale search (both no and scaled results)
python -m vidkompy align foreground.jpg background.jpg --unscaled false

# Process multiple video frames for better accuracy
python -m vidkompy align foreground.mp4 background.mp4 --num_frames 10

# Backward compatibility: 'find' command still works
python -m vidkompy find foreground.jpg background.jpg
```

**CLI Help:**

```bash
# Main command help
python -m vidkompy --help

# Video composition help
python -m vidkompy comp --help

# Thumbnail detection help (new align command)
python -m vidkompy align --help

# Backward compatibility
python -m vidkompy find --help
```

The CLI now supports these main commands:

- `comp`: Video composition with intelligent alignment
- `align`: Advanced thumbnail detection with multi-precision analysis
- `find`: Backward compatibility alias for thumbnail detection

**Video Composition Parameters:**

- `--bg`: Background video path
- `--fg`: Foreground video path
- `--output`: Output video path (auto-generated if not provided)
- `--engine`: Temporal alignment engine - 'full' (default) or 'mask'
- `--margin`: Border thickness for border matching mode (default: 8)
- `--smooth`: Enable smooth blending at frame edges
- `--spatial_precision`: Spatial alignment precision level 0-4 (0=ballpark ~1ms, 2=balanced ~25ms, 4=precise ~200ms, default: 2)
- `--unscaled`: Prefer unscaled for spatial alignment (default: True)
- `--verbose`: Enable verbose logging

**Thumbnail Detection Parameters (align command):**

- `fg`: Foreground image/video path (first positional argument)
- `bg`: Background image/video path (second positional argument)
- `--precision`: Precision level 0-4 (0=ballpark ~1ms, 2=balanced ~25ms, 4=precise ~200ms, default: 2)
- `--num_frames`: Maximum number of frames to process for videos (default: 7)
- `--unscaled`: If True, only search at 100% scale; if False, search both no and multi-scale (default: True)
- `--verbose`: Enable detailed output and debug logging

## 7. Performance

Recent updates have significantly improved `vidkompy`'s performance and accuracy:

### 7.1. Thumbnail Detection Performance (`align` module)

The modular architecture delivers excellent performance across different precision levels:

| Precision Level | Processing Time | Use Case | Algorithms Used |
| --- | --- | --- | --- |
| **Level 0 (Ballpark)** | ~1ms | Ultra-fast scale estimation | Histogram correlation only |
| **Level 1 (Coarse)** | ~10ms | Quick template matching | Parallel template matching |
| **Level 2 (Balanced)** | ~25ms | Default balanced approach | Feature + template matching |
| **Level 3 (Fine)** | ~50ms | High-quality detection | Hybrid multi-algorithm |
| **Level 4 (Precise)** | ~200ms | Maximum accuracy | Sub-pixel refinement |

**Key Performance Features:**

- **Numba JIT Optimization**: Critical functions compiled for 5-20x speed improvements
- **Parallel Processing**: ThreadPoolExecutor for concurrent scale testing
- **Intelligent Caching**: Algorithm results cached for repeated operations
- **Memory Efficiency**: Optimized data structures and streaming processing
- **Graceful Degradation**: Automatic fallbacks maintain performance when algorithms fail

### 7.2. Video Composition Performance (`comp` module)

Based on actual benchmarks with an 8-second test video (1920x1080 background, 1920x870 foreground, ~480 frames):

| Engine | Processing Time | Speed Ratio | Confidence | Notes |
| --- | --- | --- | --- | --- |
| **Full (default)** | 40.9 seconds | ~5x real-time | 1.000 (perfect) | Fastest overall with zero drift |
| **Mask** | 45.8 seconds | ~6x real-time | 1.000 (perfect) | Best for letterboxed content |

**Key Performance Insights:**

- **Full Engine**: Delivers perfect confidence scores (1.000) with ~5x real-time processing. Uses direct frame mapping which completely eliminates drift while maintaining excellent performance.

- **Mask Engine**: Slightly slower than Full engine but achieves perfect confidence. Ideal for content with black borders or letterboxing where content-focused comparison is beneficial.

### 7.3. Technical Optimizations

- **Zero Drift Design**: Both engines use sliding window constraints that enforce monotonicity by design, completely eliminating temporal drift.
- **Optimized Compositing**: Sequential frame reading instead of random access yields a **10-100x speedup** in the final composition stage.
- **Direct Pixel Comparison**: Frame comparison uses actual pixel values without information loss from hashing or compression.
- **Bidirectional Matching**: Forward and backward passes are merged for robust alignment results.
- **Efficient Memory Usage**: Both engines use streaming processing with reasonable memory footprints.

### 7.4. Choosing the Right Engine

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

## 8. Development

To contribute to `vidkompy`, set up a development environment using `hatch`.

### 8.1. Setup

1.  Clone the repository.
2.  Ensure you have `hatch` installed (`pip install hatch`).
3.  The project is managed through `hatch` environments defined in `pyproject.toml`.

### 8.2. Key Commands

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

## 9. License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
