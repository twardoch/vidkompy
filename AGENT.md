
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

- Create `PROGRESS.md` as a detailed flat plan with `[ ]` items. 
- Identify the most important TODO items, and create `TODO.md` with `[ ]` items. 
- Implement the changes. 
- Update `PROGRESS.md` and `TODO.md` as you go. 
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

Be creative, diligent, critical, relentless & funny!
</guidelines>


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

### Temporal Alignment Engines

**TLDR:** `vidkompy` offers two temporal alignment engines: **Fast** for quick processing with good results, and **Precise** for maximum accuracy with advanced drift correction. Both find the optimal "path" through time that perfectly syncs the foreground to the background.

---

Temporal alignment is the most critical and complex part of `vidkompy`. The goal is to create a mapping `FrameAlignment(fg_frame_idx, bg_frame_idx)` for every single foreground frame. `vidkompy` provides two distinct engines for this task:

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

The **Precise Engine** implements a sophisticated multi-resolution approach for maximum accuracy:

1.  **Multi-Resolution Hierarchical Alignment**:
   - Creates temporal pyramids at multiple resolutions (1/16, 1/8, 1/4, 1/2, full)
   - Performs coarse-to-fine alignment, starting at the lowest resolution
   - Each level refines the previous level's mapping
   - Applies drift correction every 100 frames

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

#### Engine Comparison

| Aspect | Fast Engine | Precise Engine |
|--------|-------------|----------------|
| **Algorithm** | Single-pass DTW with perceptual hashing | Multi-resolution hierarchical alignment |
| **Processing Time** | ~2x real-time | ~40x real-time |
| **Drift Handling** | Direct mapping (no drift) or interpolation | Active correction + keyframe anchoring |
| **Frame Extraction** | On-demand during composition | Full extraction before alignment |
| **Memory Usage** | Low (streaming) | High (all frames in memory) |
| **Accuracy** | Good, minimal drift at endpoints | Excellent, no drift throughout |
| **Best For** | Quick processing, standard videos | Critical applications, complex content |

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
        Temporal alignment engine - 'fast' (current) or 'precise' (coming soon) (default: 'fast')
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



START SPECIFICATION:
---
description: Create overview documentation for projects focused on video processing, synchronization, and alignment, particularly when dealing with foreground/background video composition and intelligent temporal matching
globs: *.py,*.md
alwaysApply: false
---


# main-overview

## Development Guidelines

- Only modify code directly relevant to the specific request. Avoid changing unrelated functionality.
- Never replace code with placeholders like `# ... rest of the processing ...`. Always include complete code.
- Break problems into smaller steps. Think through each step separately before implementing.
- Always provide a complete PLAN with REASONING based on evidence from code and logs before making changes.
- Explain your OBSERVATIONS clearly, then provide REASONING to identify the exact issue. Add console logs when needed to gather more information.


The vidkompy system implements intelligent video overlay and synchronization with three core business domains:

## Frame Analysis System (Importance: 95)
- Perceptual hashing engine combines multiple algorithms:
  - Frequency domain analysis (pHash)
  - Brightness patterns (AverageHash) 
  - Color distribution statistics (ColorMomentHash)
  - Edge detection (MarrHildrethHash)
- Weighted fingerprint generation for optimal frame matching
- Border region masking for edge-based alignment

## Temporal Alignment Engine (Importance: 98)
- Dynamic Time Warping (DTW) implementation with:
  - Sakoe-Chiba band constraints
  - Monotonic frame mapping
  - Keyframe interpolation
- Adaptive keyframe density calculation
- Foreground-first preservation principle
- Frame sequence validation

## Spatial Positioning System (Importance: 92)
- Template matching with normalized correlation
- Auto-scaling for resolution mismatches
- Border mode alignment options
- Position confidence scoring
- Center alignment fallback

## Video Composition Rules (Importance: 85)
- Foreground frame preservation guarantee
- Background frame warping/adaptation
- Audio stream selection logic
- Quality-based decision system
- Temporal drift prevention

Core Integration Points:
```
src/vidkompy/
  ├── core/
  │   ├── alignment_engine.py    # Orchestration logic
  │   ├── dtw_aligner.py        # Temporal matching
  │   ├── frame_fingerprint.py  # Frame analysis
  │   └── spatial_alignment.py  # Position detection
```

$END$
END SPECIFICATION