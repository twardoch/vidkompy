## Coding style

<guidelines>
# When you write code

- Iterate gradually, avoiding major changes
- Minimize confirmations and checks
- Preserve existing code/structure unless necessary
- Use constants over magic numbers
- Check for existing solutions in the codebase before starting
- Check often the coherence of the code you’re writing with the rest of the code.
- Focus on minimal viable increments and ship early
- Write explanatory docstrings/comments that explain what and WHY this does, explain where and how the code is used/referred to elsewhere in the code
- Analyze code line-by-line
- Handle failures gracefully with retries, fallbacks, user guidance
- Address edge cases, validate assumptions, catch errors early
- Let the computer do the work, minimize user decisions
- Reduce cognitive load, beautify code
- Modularize repeated logic into concise, single-purpose functions
- Favor flat over nested structures
- Consistently keep, document, update and consult the holistic overview mental image of the codebase.

Work in rounds:

- Create `PLAN.md` as a detailed flat plan with `[ ]` items.
- Identify the most important TODO items, and create `TODO.md` with `[ ]` items.
- Implement the changes.
- Update `PLAN.md` and `TODO.md` as you go.
- After each round of changes, update `CHANGELOG.md` with the changes.
- Update `README.md` to reflect the changes.

## Keep track of paths

In each source file, maintain the up-to-date `this_file` record that shows the path of the current file relative to project root. Place the `this_file` record near the top of the file, as a comment after the shebangs, or in the YAML Markdown frontmatter.

## When you write Python

- Use `uv pip`, never `pip`
- Use `python -m` when running code
- PEP 8: Use consistent formatting and naming
- Write clear, descriptive names for functions and variables
- PEP 20: Keep code simple and explicit. Prioritize readability over cleverness
- Use type hints in their simplest form (list, dict, | for unions)
- PEP 257: Write clear, imperative docstrings
- Use f-strings. Use structural pattern matching where appropriate
- ALWAYS add "verbose" mode logugu-based logging, & debug-log
- For CLI Python scripts, use fire & rich, and start the script with

```
#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["PKG1", "PKG2"]
# ///
# this_file: PATH_TO_CURRENT_FILE
```

Ask before extending/refactoring existing code in a way that may add complexity or break things.

When you’re finished, print "Wait, but" to go back, think & reflect, revise & improvement what you’ve done (but don’t invent functionality freely). Repeat this. But stick to the goal of "minimal viable next version". Lead two experts: "Ideot" for creative, unorthodox ideas, and "Critin" to critique flawed thinking and moderate for balanced discussions. The three of you shall illuminate knowledge with concise, beautiful responses, process methodically for clear answers, collaborate step-by-step, sharing thoughts and adapting. If errors are found, step back and focus on accuracy and progress.

## After Python changes run:

```
fd -e py -x autoflake {}; fd -e py -x pyupgrade --py312-plus {}; fd -e py -x ruff check --output-format=github --fix --unsafe-fixes {}; fd -e py -x ruff format --respect-gitignore --target-version py312 {}; python -m pytest;
```

Be creative, diligent, critical, relentless & funny! </guidelines>

# `vidkompy`

**Intelligent Video Overlay and Synchronization**

`vidkompy` is a powerful command-line tool engineered to overlay a foreground video onto a background video with exceptional precision and automatic alignment. The system intelligently handles discrepancies in resolution, frame rate, duration, and audio, prioritizing content integrity and synchronization accuracy over raw processing speed.

The core philosophy of `vidkompy` is to treat the **foreground video as the definitive source of quality and timing**. All its frames are preserved without modification or re-timing. The background video is dynamically adapted—stretched, retimed, and selectively sampled—to synchronize perfectly with every frame of the foreground content, ensuring a seamless and coherent final output.

---

## Features

### Video Composition

- **Automatic Spatial Alignment**: Intelligently detects the optimal x/y offset to position the foreground video within the background, even if they are cropped differently.
- **Advanced Temporal Synchronization**: Aligns videos with different start times, durations, and frame rates, eliminating temporal drift and ensuring content matches perfectly over time.
- **Foreground-First Principle**: Guarantees that every frame of the foreground video is included in the output, preserving its original timing and quality. The background video is adapted to match the foreground.
- **Drift-Free Alignment**: Uses optimized sliding window algorithms to create globally optimal, monotonic alignment, preventing the common "drift-and-catchup" artifacts.
- **High-Performance Processing**: Leverages multi-core processing, direct pixel comparison, and optimized video I/O to deliver results quickly.
- Sequential video composition is 10-100x faster than random-access methods.
- **Smart Audio Handling**: Automatically uses the foreground audio track if available, falling back to the background audio. The audio is correctly synchronized with the final video.
- **Flexible Operation Modes**: Supports specialized modes like `mask` for letterboxed content and `smooth` blending for seamless visual integration.

### Thumbnail Detection

- **Multi-Scale Template Matching**: Advanced thumbnail detection system for finding scaled and translated foreground images within background images/videos
- **Fast Histogram Correlation**: Initial scale estimation using histogram correlation for rapid processing
- **Dual Result Analysis**: Provides both unity scale and scaled options with confidence metrics
- **Precision Control**: Configurable precision parameter for varying levels of analysis detail
- **Rich Output**: Detailed match results with confidence scores and processing statistics

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

| Aspect         | Full                    | Mask                            |
| -------------- | ----------------------- | ------------------------------- |
| **Algorithm**  | Direct pixel comparison | Masked pixel comparison         |
| **Speed**      | ~5x real-time           | ~5x real-time                   |
| **Drift**      | Zero (monotonic)        | Zero (monotonic)                |
| **Memory**     | Medium                  | Medium                          |
| **Confidence** | Perfect (1.000)         | Perfect (1.000)                 |
| **Best For**   | Standard videos         | Letterboxed/pillarboxed content |

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

`vidkompy` now offers two main commands: video composition and thumbnail detection.

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
```

**Thumbnail Detection Examples:**

```bash
# Find thumbnail in image
python -m vidkompy find background.jpg foreground.jpg

# Find thumbnail in video frame
python -m vidkompy find background.mp4 foreground.jpg

# High precision analysis
python -m vidkompy find background.jpg foreground.jpg --precision 10

# Detailed output with processing time
python -m vidkompy find background.jpg foreground.jpg --verbose
```

**CLI Help:**

```bash
# Main command help
python -m vidkompy --help

# Video composition help
python -m vidkompy comp --help

# Thumbnail detection help
python -m vidkompy find --help
```

The CLI now supports two main commands:

- `comp`: Video composition with intelligent alignment
- `find`: Thumbnail detection in images/videos

**Video Composition Parameters:**

- `--bg`: Background video path
- `--fg`: Foreground video path
- `--output`: Output video path (auto-generated if not provided)
- `--engine`: Temporal alignment engine - 'full' (default) or 'mask'
- `--margin`: Border thickness for border matching mode (default: 8)
- `--smooth`: Enable smooth blending at frame edges
- `--verbose`: Enable verbose logging

**Thumbnail Detection Parameters:**

- `background`: Background image/video path
- `foreground`: Foreground image path
- `--precision`: Analysis precision level (1-10, default: 3)
- `--verbose`: Enable detailed output

## Performance

Recent updates have significantly improved `vidkompy`'s performance and accuracy:

### Real-World Performance Comparison

Based on actual benchmarks with an 8-second test video (1920x1080 background, 1920x870 foreground, ~480 frames):

| Engine | Processing Time | Speed Ratio | Confidence | Notes |
| --- | --- | --- | --- | --- |
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

## START SPECIFICATION:

description: This rule applies when documenting the high-level business logic and domain-specific implementation of vidkompy, an intelligent video overlay and synchronization system. It focuses on core algorithmic components and unique technical approaches. globs: _.py,_.md alwaysApply: false

---

# main-overview

## Development Guidelines

- Only modify code directly relevant to the specific request. Avoid changing unrelated functionality.
- Never replace code with placeholders like `# ... rest of the processing ...`. Always include complete code.
- Break problems into smaller steps. Think through each step separately before implementing.
- Always provide a complete PLAN with REASONING based on evidence from code and logs before making changes.
- Explain your OBSERVATIONS clearly, then provide REASONING to identify the exact issue. Add console logs when needed to gather more information.

# Intelligent Video Overlay System

## Core Business Components

### Frame Fingerprinting System

- Multi-algorithm perceptual hash fusion combining:
  - Frequency domain analysis (DCT-based pHash)
  - Average intensity patterns (AverageHash)
  - Color distribution statistics (ColorMomentHash)
  - Edge structure detection (MarrHildrethHash)
- Weighted fingerprint generation with hybrid similarity scoring
- Optimized for video frame sequence comparison

### Temporal Alignment Engine

Two specialized engines with distinct approaches:

1. Full Engine (Default)

- Direct pixel-level frame comparison
- Bidirectional matching with forward/backward passes
- Sliding window constraints for monotonic progression
- Zero temporal drift by design

2. Mask Engine

- Content-aware masking for letterboxed content
- Automatic non-black region detection
- Masked frame comparison while maintaining temporal constraints
- Specialized for videos with varying aspect ratios

### Spatial Overlay System

- Template matching with normalized cross-correlation
- Intelligent scale detection and adjustment
- Optimal positioning determination
- Border region analysis for edge alignment

### Thumbnail Detection

- Multi-scale progressive refinement system
- Feature-based alignment with ORB detection
- Scale/translation parameter extraction
- Transformation validation with confidence scoring

## Key Business Rules

1. Foreground Video Integrity

- All foreground frames preserved without modification
- Background frames dynamically adapted to match
- Foreground timing treated as source of truth

2. Temporal Synchronization

- One-to-one frame mapping requirement
- Strict monotonic progression enforcement
- No temporal drift allowed
- Keyframe-based synchronization anchoring

3. Quality Preservation

- Foreground quality never compromised
- Background adaptations maintain visual coherence
- Intelligent audio track selection and sync
- Content-aware masking for aspect ratio differences

$END$ END SPECIFICATION
