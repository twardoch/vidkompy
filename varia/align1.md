# Thumbnail Finder

**Intelligent detection of scaled and translated thumbnails within images and videos**

Thumbnail Finder is a high-precision Python tool that detects where a foreground image appears as a scaled-down thumbnail within a background image or video. It solves the inverse problem of outpainting and rescaling: given an original image and a processed output containing a thumbnail of that image, find the exact scale factor and position of the thumbnail.

## Features

- **Dual Algorithm Approach**: Combines multi-scale template matching with feature-based detection for maximum accuracy
- **Multi-format Support**: Works with both images (PNG, JPG, BMP, TIFF) and videos (MP4, AVI, MOV, MKV, etc.)
- **Pixel-perfect Accuracy**: Returns integer pixel coordinates and precise scale percentages
- **Multi-frame Robustness**: Processes multiple frames and averages results for improved reliability
- **Performance Optimized**: Uses Numba JIT compilation and OpenCV's optimized algorithms
- **Rich CLI Interface**: Beautiful command-line output with progress tracking and formatted results

## Installation

### Prerequisites

- Python 3.8+
- uv package manager (recommended) or pip

### Install Dependencies

The script uses inline dependency specification, so you can run it directly with `uv`:

```bash
# Using uv (recommended)
./align.py --help

# Or install dependencies manually
pip install opencv-python numpy fire rich pathlib numba
```

## Usage

### Basic Usage

```bash
# Find thumbnail in images
./align.py --fg original.jpg --bg output_with_thumbnail.jpg

# Process videos (extracts up to 7 frames by default)
./align.py --fg input_video.mp4 --bg output_video.mp4

# Custom frame count and verbose output
./align.py --fg input.mp4 --bg output.mp4 --frames 10 --verbose
```

### Command Line Arguments

- `--fg` (required): Path to foreground image/video (the original input)
- `--bg` (required): Path to background image/video (contains the thumbnail)
- `--frames` (optional): Maximum number of frames to extract from videos (default: 7)
- `--verbose` (optional): Enable detailed logging and progress information

### Example Output

```
Thumbnail Detection Results
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric                       ┃ Value     ┃ Unit   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
│ Confidence                   │ 89.45     │ %      │
│ Scale (FG → thumbnail)       │ 75.20     │ %      │
│ X shift (thumbnail in BG)    │ 125       │ px     │
│ Y shift (thumbnail in BG)    │ 200       │ px     │
│ Scale (BG → FG size)         │ 133.00    │ %      │
│ X shift (FG on upscaled BG)  │ -166      │ px     │
│ Y shift (FG on upscaled BG)  │ -266      │ px     │
└──────────────────────────────┴───────────┴────────┘

Summary:
Confidence: 89.45%
Scale down: 75.20%
Position: (125, 200) px
Scale up: 133.00%
Reverse position: (-166, -266) px
```

## How It Works

### Problem Description

Given:

- **Foreground (FG)**: Original image/video frames (e.g., 1920×1080)
- **Background (BG)**: Processed output containing a thumbnail of FG (e.g., 1920×1440)

Find:

- Scale factor to transform FG into the thumbnail
- (x, y) position where the thumbnail appears in BG
- Reverse transformation parameters

### Algorithm Overview

1. **Frame Extraction**:

   - For images: Load directly
   - For videos: Extract the first n frames from the beginning

2. **Multi-Scale Template Matching**:

   - Test scale factors from 10% to 100% in 50 steps
   - Use normalized cross-correlation for robustness to lighting changes
   - Convert to grayscale for faster processing

3. **Feature-Based Backup**:

   - Use ORB (Oriented FAST and Rotated BRIEF) feature detection
   - Match keypoints between FG and BG frames
   - Estimate transformation using RANSAC-based homography

4. **Multi-Frame Averaging**:

   - Process all FG×BG frame combinations
   - Weight results by confidence scores
   - Average similar results for increased robustness

5. **Result Calculation**:
   - Forward transformation: FG → thumbnail in BG
   - Reverse transformation: BG upscaled to match FG size

### Technical Details

- **Template Matching**: Uses OpenCV's `TM_CCOEFF_NORMED` method for scale and illumination invariance
- **Feature Detection**: ORB features with FLANN-based matching and Lowe's ratio test
- **Performance**: Numba JIT compilation for critical numerical computations
- **Robustness**: Automatic fallback from template matching to feature matching when confidence is low

## Use Cases

- **Video Processing**: Find where original content appears in processed/edited videos
- **Image Forensics**: Detect scaled copies within larger compositions
- **Quality Control**: Verify correct thumbnail placement in automated workflows
- **Reverse Engineering**: Understand transformations applied to images/videos

## Performance

- **Speed**: Processes typical image pairs in 1-5 seconds
- **Accuracy**: Pixel-perfect localization with sub-percent scale precision
- **Robustness**: Handles lighting changes, compression artifacts, and minor content variations
- **Scalability**: Efficiently processes multiple frames with parallel-optimized algorithms

## Limitations

- Assumes uniform scaling (same factor for width and height)
- No rotation detection (only scale and translation)
- Requires sufficient visual features for feature-based matching
- Performance decreases with very large images or many frames

## Algorithm Background

The implementation is based on extensive research into multi-scale template matching and feature-based image registration. Key techniques include:

- **Multi-Scale Template Matching**: Systematic scale-space search with normalized correlation
- **Scale-Invariant Features**: ORB detector for robust keypoint matching across scales
- **RANSAC Estimation**: Outlier-robust transformation parameter estimation
- **Weighted Averaging**: Confidence-based combination of multiple measurements

## Contributing

This tool is part of the `vidkompy` project. See the main project documentation for contribution guidelines.

## License

MIT License - see the main `vidkompy` project for details.
