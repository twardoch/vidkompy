# vidkompy: Intelligent Video Overlay Tool

## 1. Lights, Camera, Align! The Story of vidkompy

Ever tried to overlay one video onto another, only to find yourself lost in a frustrating game of nudge-the-timeline and eyeball-the-pixels? You're not alone. Synchronizing two video sources, especially when they have different start times, frame rates, resolutions, or even slight content variations, can be a real headache.

**vidkompy (`vidkompy.py`) is here to change that.**

This command-line tool isn't just about slapping one video on top of another. It's about doing it *intelligently*. Born from the need for a robust solution to automatically align and composite videos, vidkompy dives deep into the content of your footage—analyzing audio, visual features, and frame sequences—to find the best possible match.

Our core philosophy? **Precision and intelligence over raw speed.** While vidkompy aims to be efficient, its primary goal is to produce a high-quality, seamlessly aligned output, minimizing manual intervention and guesswork.

## 2. Under the Hood: Architecture and Magic

vidkompy is a Python script that orchestrates several powerful open-source libraries to achieve its magic:

*   **`ffmpeg-python`**: The workhorse for all heavy-lifting video and audio operations. It's used for probing video metadata, extracting audio streams, and, importantly, for the final composition of video and audio tracks into the output file.
*   **`OpenCV (cv2)`**: Your friendly neighborhood computer vision expert. OpenCV is crucial for:
    *   **Spatial Alignment**: Figuring out *where* to place the foreground video on the background. It does this by analyzing frame content using template matching or feature detection.
    *   **Frame-Level Analysis**: For the advanced `frames` temporal alignment mode, OpenCV helps in extracting individual frames, comparing them for similarity (using SSIM from `scikit-image`), and enabling precise frame-by-frame composition.
*   **`SciPy` & `NumPy`**: The numerical powerhouses. `SciPy`'s signal processing capabilities are used for audio cross-correlation (a key technique in the `audio` temporal alignment mode), while `NumPy` handles the array manipulations essential for image and audio data.
*   **`soundfile`**: Used for reading audio data from temporary WAV files extracted by FFmpeg.
*   **`Loguru`**: For clear, beautiful, and informative logging. Helps you understand what vidkompy is thinking, especially with the `--verbose` flag.
*   **`Fire`**: Transforms the Python script into a user-friendly command-line interface.
*   **`Rich`**: Provides elegant progress bars and console output, making the process more pleasant to watch.

**The General Workflow:**

1.  **Analysis**: vidkompy starts by probing both background (`--bg`) and foreground (`--fg`) videos to understand their properties: resolution, FPS, duration, audio presence, etc.
2.  **Temporal Alignment**: Next, it determines the best time offset for the foreground video relative to the background. This is a critical step and offers multiple strategies (see below).
3.  **Spatial Alignment**: Once the temporal aspect is known (or an initial estimate is made), vidkompy figures out the (x, y) coordinates to place the foreground video on the background.
4.  **Composition**: Finally, armed with temporal and spatial offsets, vidkompy composites the videos. Depending on the chosen temporal alignment mode, this might involve a sophisticated frame-by-frame composition using OpenCV or a more straightforward FFmpeg overlay command. Audio is carefully selected and merged.

## 3. Finding the Perfect Timing: Temporal Alignment Deep Dive

Getting the timing right is paramount. vidkompy offers three distinct methods for temporal alignment, selectable with the `--temporal_align` option:

### 3.1. `--temporal_align audio` (The Maestro's Choice)

*   **How it Works**: If both your background and foreground videos have audio tracks that are expected to correspond (e.g., recordings of the same event), this mode is often your best bet.
    1.  vidkompy extracts audio from both videos into temporary WAV files (mono, 16kHz for efficiency).
    2.  It then performs a **cross-correlation** on these audio signals using `scipy.signal.correlate`. Imagine sliding one audio waveform across the other and, at each point, measuring how similar they are.
    3.  The point where the similarity score (correlation) is highest indicates the optimal time offset. This offset tells vidkompy how much the foreground video needs to be shifted to align its audio (and thus, its content) with the background video.
*   **Benefits**: Highly effective for synchronizing footage from different cameras or takes of the same scene, especially if there are distinct sounds (claps, speech, music beats).
*   **Limitations**: Requires both videos to have reasonably similar audio content. If one is silent or the audio is completely unrelated, this mode will likely fall back or produce a zero offset. Very noisy audio can also reduce accuracy.
*   **Behind the Scenes**: The tool logs the computed offset and a "confidence" score (peak correlation vs. average) to give you an idea of how strong the match was.

### 3.2. `--temporal_align duration` (The Simple Synchronizer)

*   **How it Works**: This is the most straightforward approach. If the background video is longer than the foreground, it calculates an offset to center the foreground video within the duration of the background. If the foreground is longer, the offset is typically zero (foreground starts with the background).
*   **When to Use**: Useful when there's no common audio, or you simply want the foreground to appear for its full duration somewhere in the middle of a longer background. Also serves as a fallback if other modes can't find a confident alignment.

### 3.3. `--temporal_align frames` (The Pixel-Perfect Perfectionist) - **Recommended for Highest Quality Sync**

This is vidkompy's most advanced and computationally intensive mode, designed to tackle tricky desynchronization issues, slight speed variations, or when you need the absolute best possible frame-to-frame content alignment. It's where the "intelligent" part of vidkompy truly shines.

*   **The Challenge**: Videos aren't always perfectly stable. Recording devices might have slightly different clock speeds, leading to drift. Content itself might have pauses or speed ramps. A single time offset won't fix "repeating desyncs" where videos align, then drift, then re-align.
*   **The `frames` Mode Solution**:
    1.  **Keyframe Sampling**: vidkompy first extracts a series of sample frames from both videos (e.g., 2 frames per second). This creates a manageable set of "keyframes" for comparison.
    2.  **Frame Similarity (SSIM)**: For each sample frame from the foreground video, it searches for the most similar-looking sample frame in the background video. Similarity is measured using the Structural Similarity Index (SSIM) via `skimage.metrics.structural_similarity`, which is good at perceiving visual similarity like the human eye.
    3.  **Keyframe Matching & Filtering**: This process yields a list of potential `(background_keyframe_index, foreground_keyframe_index)` pairs. These matches are then sorted and filtered to ensure they are *monotonic*—meaning, as the foreground video progresses, the corresponding matched background frames also progress generally forward in time. This helps eliminate erroneous matches.
    4.  **Dynamic Frame-to-Frame Mapping**: This is the core of the magic. Using the reliable (monotonic) keyframe matches as anchors, vidkompy interpolates a *specific background frame index for every single foreground frame*. If you have keyframe matches `(FG10 -> BG50)` and `(FG20 -> BG120)`, then `FG15` would be mapped to a background frame around `BG85`. This dynamic map effectively stretches or compresses the perceived timeline between keyframes, adapting to local speed differences or drifts.
    5.  **OpenCV Composition**: With this precise frame-to-frame map, vidkompy then switches to an OpenCV-based composition pipeline (`compose_videos_opencv`):
        *   It iterates through each foreground frame specified in the alignment map.
        *   For each foreground frame, it fetches its mapped background frame.
        *   It performs the spatial overlay (see next section) of these two specific frames.
        *   The resulting composited frame is written to a temporary video file. This step produces a *silent* video.
    6.  **Synchronized Audio Merging**: After the silent, frame-perfect video is created, vidkompy uses FFmpeg (`_merge_audio_with_ffmpeg`) to add the audio back. The audio (usually from the background source) is carefully offset so that it starts in perfect sync with the content of the composited video segment. The offset is determined from the timing of the very first frame in the dynamic alignment map.
*   **The `trim` Power-Up**: When `--trim` is used with `--temporal_align frames`, the dynamic frame-to-frame mapping is *only* generated for the segment of the foreground video that lies *between the first and last successfully matched keyframes*. This is incredibly powerful: it automatically isolates and renders only the portion of the videos where a confident content match exists, effectively trimming away any leading or trailing parts that don't align. This is ideal for handling videos where only a sub-segment is relevant for overlay.
*   **Benefits**: Provides the most accurate temporal synchronization, especially for challenging videos with drift or varying content speeds. The `trim` option makes it excellent for extracting precisely aligned segments.
*   **Trade-offs**: This mode is the most computationally intensive due to frame sampling, repeated similarity comparisons, and OpenCV-based frame-by-frame writing.

## 4. Finding the Right Spot: Spatial Alignment Deep Dive

Once vidkompy knows *when* the foreground video should appear, it needs to figure out *where* on the background frame it should be placed. This is handled by the `--spatial_method` option.

*   **Initial Step**: For spatial alignment, vidkompy typically takes a sample frame from the foreground (e.g., 1 second into its content) and a corresponding frame from the background (e.g., 1 second + temporal_offset). These two frames are then used for comparison.

### 4.1. `--spatial_method template` (The Exact Matcher)

*   **How it Works**: This method assumes the foreground video frame is essentially a smaller, exact (or very close) rectangular sub-region of the background frame. It uses OpenCV's `cv2.matchTemplate` function with a normalized correlation coefficient (`TM_CCOEFF_NORMED`).
    *   Imagine sliding the foreground frame like a "template" over the background frame.
    *   At each position, OpenCV calculates how well the template matches that part of the background.
    *   The position with the highest similarity score is chosen as the top-left (x, y) coordinate for the overlay.
*   **Best For**: Ideal when the foreground is literally a crop or a scaled-down version of a portion of the background scene (e.g., a picture-in-picture effect where the smaller picture is a zoom-in of the larger one).
*   **Confidence**: vidkompy checks the confidence score of the match. If it's too low (e.g., below 0.7), it means a good template match wasn't found, and it will fall back to a default placement (usually centering).

### 4.2. `--spatial_method feature` (The Smart Detective)

*   **How it Works**: If the foreground isn't an exact sub-region—perhaps there are slight changes in angle, lighting, or scale—template matching might fail. Feature matching is more robust.
    *   vidkompy uses the ORB (Oriented FAST and Rotated BRIEF) feature detector from OpenCV (`cv2.ORB_create`). ORB finds distinctive keypoints (corners, unique patterns) in both the background and foreground sample frames and computes descriptors for them.
    *   These descriptors are then matched using a Brute-Force Matcher (`cv2.BFMatcher`) to find corresponding keypoints between the two frames.
    *   From a set of good matches, vidkompy calculates the median displacement (offset) between the matched points. This median offset is used as the (x, y) for placing the foreground.
*   **Best For**: More resilient to minor variations in scale, rotation, and illumination between the foreground and background content.
*   **Robustness**: Using the median offset helps to ignore outlier matches and provides a more stable alignment. If too few feature matches are found, it also falls back to default placement.

### 4.3. Default/Fallback and Scaling

*   **Centering**: If neither `template` nor `feature` matching yields a confident result, or if `--skip_spatial_align` is used, vidkompy defaults to placing the foreground video in the center of the background video.
*   **Foreground Scaling**: If the foreground video's dimensions are *larger* than the background's, vidkompy will automatically scale down the foreground (preserving its aspect ratio) to fit within the background dimensions before overlaying. This ensures the foreground content isn't awkwardly cropped.

## 5. Using vidkompy: Options and Examples

vidkompy is controlled via command-line arguments.

### 5.1. Core Arguments:

*   `--bg BG_VIDEO_PATH`: Path to the background video file. (Required)
*   `--fg FG_VIDEO_PATH`: Path to the foreground video file. (Required)
*   `--output OUTPUT_VIDEO_PATH`: Path for the final composited video. If not provided, a name is automatically generated (e.g., `bg_overlay_fg.mp4`).

### 5.2. Key Alignment Options:

*   `--temporal_align {audio|duration|frames}`:
    *   `audio` (default): Aligns based on audio cross-correlation.
    *   `duration`: Aligns based on centering durations.
    *   `frames`: Performs advanced frame-by-frame content matching and dynamic mapping. **Recommended for highest sync quality, especially with `--trim`**.
*   `--spatial_method {template|feature}`:
    *   `template` (default): Uses template matching for spatial placement.
    *   `feature`: Uses ORB feature matching for spatial placement.
*   `--trim`: (Boolean flag, default `False`)
    *   When used with `--temporal_align frames`, this is highly effective. It restricts the output to only the segment where confident frame-to-frame matches were found between the videos.
    *   For other temporal modes, it can trim the start of the background if a positive temporal offset is found, aligning the start of the effective overlay.
*   `--skip_spatial_align`: (Boolean flag, default `False`)
    *   If set, skips content-based spatial alignment and defaults to centering the foreground on the background.

### 5.3. Other Options:

*   `--verbose`: (Boolean flag, default `False`)
    *   Enables detailed debug logging to the console, showing more insight into the alignment process and decisions. Highly recommended if you're troubleshooting or curious.

### 5.4. Example Usage:

**1. Basic Overlay (Defaults: audio temporal, template spatial):**

```bash
./vidkompy.py --bg background.mp4 --fg foreground.mov -o result.mp4
```

**2. Highest Quality Frame-Level Sync, Trimmed to Matched Segment:**

```bash
./vidkompy.py --bg scene_cam1.mp4 --fg scene_cam2.mp4 \
    --temporal_align frames --trim \
    --spatial_method feature \
    -o synced_segment.mp4 --verbose
```
*(This is often the best combination for tricky synchronization where you want the most precisely aligned portion of two clips.)*

**3. Align by Audio, but use Feature-Based Spatial Placement:**

```bash
./vidkompy.py --bg main_shot.mp4 --fg overlay_graphic_synced_to_audio.mov \
    --temporal_align audio \
    --spatial_method feature \
    -o audio_synced_features.mp4
```

**4. Center Foreground in Background by Duration (No Audio/Frame Sync):**

```bash
./vidkompy.py --bg long_ambience.mp4 --fg short_clip.mp4 \
    --temporal_align duration \
    --skip_spatial_align \
    -o centered_clip.mp4
```

## 6. Aiming for Excellence: Output Quality

vidkompy strives for high-quality output:

*   **Frame Rate**: The output video will use the *higher* of the two input frame rates, preserving motion smoothness.
*   **Duration**: Typically, the output will cover the *longer* of the two video durations (after alignment). If the foreground ends, the background continues (or vice-versa if the background was padded in the `frames` mode processing, though this is less common). With `trim` in `frames` mode, the duration is that of the aligned segment.
*   **Video Encoding**: By default, videos are encoded using `libx264` (a high-quality H.264 encoder) with a CRF (Constant Rate Factor) of 18 and the `medium` preset. This provides an excellent balance of quality and file size. The pixel format is `yuv420p` for broad compatibility.
*   **Audio Encoding**: Audio is typically encoded to AAC at 192kbps. If using the background audio, and it's already in a compatible format, vidkompy might copy it directly to preserve quality (though the default is re-encoding to AAC for consistency).
*   **Audio Selection**:
    *   If both videos have audio, the background video's audio is prioritized for the final output (after using both for alignment if in `audio` mode).
    *   If only one video has audio, that audio track is used.
    *   If neither has audio, the output is silent.

## 7. The Journey So Far & The Road Ahead

vidkompy has evolved from a basic concept to a sophisticated tool capable of handling complex alignment scenarios. The `frames` temporal alignment mode, in particular, represents a significant step towards true content-aware synchronization.

**Current Strengths:**

*   Multiple robust temporal and spatial alignment strategies.
*   Intelligent frame-by-frame mapping for the `frames` mode.
*   Effective `trim` functionality, especially powerful with `frames` mode.
*   Clear logging and progress indication.

**Potential Future Enhancements (Ideas from the original spec):**

*   More advanced audio analysis (e.g., spectral flatness, DTW for audio if extreme non-linearities).
*   Visual-based temporal sync (e.g., detecting flashes or specific motion events if audio is unavailable).
*   User-defined alignment points or manual offset overrides.
*   Support for multiple foreground overlays.
*   More sophisticated handling of significant scale/rotation/perspective differences in spatial alignment (e.g., full homography).

## 8. Installation & Dependencies

vidkompy is a Python script. To run it, you'll need Python 3 (preferably 3.10+) and the following packages, which are listed in the script's header and can be installed using `uv pip` (or `pip`):

```
# /// script
# dependencies = ["fire", "rich", "loguru", "opencv-python", "numpy", "scipy", "ffmpeg-python", "soundfile", "scikit-image"]
# ///
```

You can install them via:

```bash
uv pip install fire rich loguru opencv-python numpy scipy ffmpeg-python soundfile scikit-image
```

You also need **FFmpeg** installed and accessible in your system's PATH, as `ffmpeg-python` is a wrapper around the FFmpeg command-line tool.

---

We hope vidkompy helps you conquer your video overlay challenges with intelligence and precision! Happy editing! 