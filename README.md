# vidkompy

`vidkompy` is an intelligent command-line tool designed to overlay a foreground video onto a background video with high-precision automatic alignment. The system is engineered to handle differences in resolution, frame rate, duration, and audio, prioritizing content integrity and accuracy over raw processing speed.

The core philosophy is to treat the **foreground video as the definitive source of quality and timing**. All its frames are preserved without modification or retiming. The background video is dynamically adapted—stretched, retimed, and selectively sampled—to synchronize perfectly with the foreground content.

## How It Works: The Processing Pipeline

The alignment and composition process is orchestrated by the `AlignmentEngine` and follows a meticulous, multi-stage pipeline:

1.  **Video Analysis**: The process begins by probing both background and foreground videos using `ffprobe` to extract essential metadata. This includes resolution, frames per second (FPS), duration, frame count, and audio presence. This information is used to check for compatibility and inform subsequent alignment decisions.

2.  **Spatial Alignment**: The engine determines the optimal (x, y) coordinates to place the foreground video on top of the background. A sample frame from the middle of each video is used for this calculation. The system then calculates the offset needed to position the foreground video correctly within the background frame.

3.  **Temporal Alignment**: This is the most critical phase, where the videos are synchronized in time. `vidkompy` finds the perfect background frame to match _every single_ foreground frame, ensuring the foreground's timing is flawlessly preserved. This creates a detailed frame-by-frame mapping.

4.  **Video Composition**: With alignment data computed, the final video is composed:

- First, a silent video is created using OpenCV. The engine iterates through the frame alignment map, reads the corresponding frames from each video, overlays them using the spatial offset, and writes the composite frame to a temporary file. The output FPS is set to the foreground video's FPS to ensure no foreground frames are dropped.
- Next, an audio track is added using FFmpeg. The tool intelligently selects the audio source, **prioritizing the foreground video's audio**. The audio is synchronized with the composed video, and the final output file is generated.

## Alignment Methods in Detail

`vidkompy` employs sophisticated algorithms for both temporal and spatial alignment, with different modes to balance speed and precision.

### Temporal Alignment: Finding the Perfect Sync

Temporal alignment synchronizes the two videos over time. This can be done using audio cues or by analyzing visual content.

#### **`--match_time fast`**: This mode first attempts to align the videos using their audio tracks, which is very fast and efficient.

- **Audio-Based Synchronization**: The audio from both videos is extracted into temporary WAV files. The system then computes the **cross-correlation** of the two audio signals. The peak of the correlation reveals the time offset needed to sync the tracks. A confidence score is calculated to validate the match. If audio alignment succeeds, a simple frame mapping is created based on this single time offset.
- **Fallback**: If audio is unavailable in either video or the extraction fails, this mode automatically falls back to the more intensive frame-based alignment method.

#### **`--match_time precise`** (Default): This mode goes directly to frame-based alignment for the most accurate synchronization, which is essential for videos without clear audio cues or with slight timing drifts.

- **1. Keyframe Selection**: To work efficiently, the aligner samples a limited number of frames (keyframes) from both videos. It samples the foreground video based on a configurable `max_keyframes` limit and then samples the background more densely to ensure a good search space.
- **2. Keyframe Matching**: For each foreground keyframe, the system searches for the best-matching background keyframe. The similarity between frame pairs is calculated using the **Structural Similarity Index (SSIM)**. A match is only accepted if the similarity score exceeds a confidence threshold of 0.6.
- **3. Monotonic Filtering**: The list of keyframe matches is filtered to ensure a logical time progression. Any match where a later foreground frame corresponds to an earlier background frame is discarded to prevent temporal inconsistencies.
- **4. Frame Interpolation**: With a clean set of keyframe matches, the engine builds a complete alignment map for _every single foreground frame_. For frames that fall between keyframes, the corresponding background frame index is calculated using **smooth interpolation** (a smoothstep function). This is a crucial step that ensures the background video's motion appears natural and fluid, avoiding the jerky "drift-and-catchup" issues that can plague simpler methods.

### Spatial Alignment: Finding the Perfect Position

Spatial alignment determines where the foreground video sits within the larger background frame. If the foreground video is larger than the background, it is automatically scaled down to fit while preserving its aspect ratio.

#### ** `--match_space precise` / `template` ** (Default): This method uses **Template Matching**.

- It treats the entire foreground frame as a template and searches for its most likely position within the background frame.
- The match is performed using normalized cross-correlation (`cv2.TM_CCOEFF_NORMED`), which is highly precise for finding exact pixel-level positions. A confidence score is generated based on the quality of the match.

#### ** `--match_space fast` / `feature` **: This method uses **Feature Matching**, which is more robust against slight variations like compression or color changes.

- It uses the **ORB (Oriented FAST and Rotated BRIEF)** algorithm to detect hundreds of key feature points in both frames.
- It then matches these keypoints between the two frames and calculates the median displacement to find the overlay coordinates.
- If too few features are found or matched, it automatically falls back to a safe center alignment.

#### **`--skip_spatial_align`**: If this flag is used, all alignment calculations are skipped, and the foreground video is simply centered within the background frame.

## Usage

### Prerequisites

You must have the **FFmpeg** binary installed on your system and accessible in your system's PATH. `vidkompy` depends on it for all video and audio processing tasks.

### Installation

The tool is a Python package and can be installed using pip:

```bash
uv pip install .
```

### Command-Line Interface (CLI)

The tool is run from the command line. The primary arguments are the paths to the background and foreground videos.

```bash
python -m vidkompy --bg <background_video> --fg <foreground_video> [OPTIONS]
```

**Key Arguments:**

- `--bg` (str): Path to the background video file. **[Required]**
- `--fg` (str): Path to the foreground video file. **[Required]**
- `-o`, `--output` (str): Path for the final output video. If not provided, a name is automatically generated (e.g., `bg-stem_overlay_fg-stem.mp4`).
- `--match_time` (str): The temporal alignment method.
  - `'precise'` (default): Uses frame-based matching for maximum accuracy.
  - `'fast'`: Attempts audio-based alignment first and falls back to frames.
- `--match_space` (str): The spatial alignment method.
  - `'precise'` or `'template'` (default): Slower but more exact pixel matching.
  - `'fast'` or `'feature'`: Faster, more robust feature-based matching.
- `--trim` (bool): If `True` (default), the output video is trimmed to the duration of the aligned segment, dropping any un-matched frames from the beginning or end.
- `--skip_spatial_align` (bool): If `True` (default: `False`), skips spatial alignment and centers the foreground video.
- `--verbose` (bool): Enables detailed debug logging for troubleshooting.
- `--max_keyframes` (int): Sets the maximum number of keyframes to use for the `'precise'` temporal alignment mode (default: 2000).
