# SPEC5: vidkompy Performance and Drift Elimination Improvements

## Executive Summary

The current implementation suffers from two critical issues:
1. **Temporal Drift**: Sparse keyframe interpolation causes background video to drift and "catch up" periodically
2. **Performance**: SSIM-based matching is extremely slow (3.5 minutes for 8-second videos)

This spec outlines a comprehensive solution that addresses both issues through algorithmic improvements and architectural optimizations.

## Problem Analysis

### 1. Temporal Drift Root Cause

The drift occurs because:
- Sparse keyframe sampling (e.g., 12 keyframes for 483 frames) creates large interpolation gaps
- Linear interpolation between keyframes assumes constant speed, but videos rarely have perfectly uniform timing
- Small timing differences accumulate over interpolation spans
- The "catch-up" effect happens when the next keyframe forces a correction

### 2. Performance Bottlenecks

From the logs:
- Building cost matrix took ~3 minutes for just 14×24 comparisons
- Each SSIM computation requires frame extraction, resizing, and pixel-by-pixel comparison
- Border mode forces SSIM usage even when perceptual hashing is available
- No parallelization of independent comparisons

## Proposed Solutions

### Phase 1: Eliminate Drift with Dense Matching (Priority: CRITICAL)

#### 1.1 Adaptive Keyframe Density
```python
def calculate_adaptive_keyframe_density(video_info: VideoInfo, target_drift_ms: float = 16.67):
    """
    Calculate keyframe density to ensure drift never exceeds target.
    Default 16.67ms = 1 frame at 60fps
    """
    # Sample every N frames where N ensures interpolation error < target
    max_interpolation_span = int(target_drift_ms * video_info.fps / 1000)
    return max(1, min(max_interpolation_span, video_info.frame_count // 100))
```

#### 1.2 Sliding Window Refinement
After initial keyframe matching, add a refinement pass:
```python
def refine_alignment_with_sliding_window(
    initial_matches: List[FrameAlignment],
    window_size: int = 30,
    step: int = 10
):
    """
    Refine alignment by checking small windows between keyframes.
    Catches drift before it accumulates.
    """
    # For each gap between keyframes
    # Sample and verify interpolation accuracy
    # Add correction points where drift detected
```

#### 1.3 Implement Proper DTW for Border Mode
The current implementation disables DTW in border mode. Fix this:
```python
def compute_masked_fingerprint(frame: np.ndarray, mask: np.ndarray) -> dict:
    """
    Compute fingerprint only for masked regions.
    Enables DTW with border matching.
    """
    # Apply mask before computing perceptual hash
    # Use multiple hash types for robustness
    # Weight by visible pixel count
```

### Phase 2: Performance Optimization (Priority: HIGH)

#### 2.1 Hierarchical Matching
```python
def hierarchical_temporal_alignment(bg_info, fg_info):
    """
    Multi-resolution temporal alignment:
    1. Coarse alignment at 1/8 resolution, every 8th frame
    2. Refine at 1/4 resolution around coarse matches  
    3. Final alignment at 1/2 resolution for precision
    """
    # Dramatically reduces comparisons while maintaining accuracy
```

#### 2.2 Perceptual Hash Optimization
```python
class OptimizedFingerprinter:
    def __init__(self):
        # Use only the fastest, most discriminative hashes
        self.primary_hash = cv2.img_hash.PHash_create()  # Best quality/speed
        self.quick_hash = cv2.img_hash.AverageHash_create()  # For pre-filtering
        
    def batch_compute_gpu(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Use GPU acceleration if available."""
        # Batch process multiple frames
        # Utilize CUDA for hash computation
```

#### 2.3 Parallel Cost Matrix Construction
```python
def build_cost_matrix_parallel(bg_indices, fg_indices, n_workers=cpu_count()):
    """
    Parallelize independent similarity computations.
    """
    # Divide matrix into chunks
    # Process chunks in parallel
    # Merge results
```

### Phase 3: Architectural Improvements (Priority: MEDIUM)

#### 3.1 Caching Strategy
```python
class AlignmentCache:
    """
    Cache computed alignments for common video pairs.
    Especially useful for iterative workflows.
    """
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        
    def get_cache_key(self, bg_info, fg_info, options) -> str:
        # Hash of video properties and alignment options
        
    def save_alignment(self, alignment: TemporalAlignment, key: str):
        # Persist to disk with metadata
```

#### 3.2 Progressive Alignment
```python
class ProgressiveAligner:
    """
    Start with fast approximate alignment, refine progressively.
    User sees results quickly, quality improves in background.
    """
    def align_progressive(self, bg_info, fg_info, callback):
        # Quick pass: Every 30th frame
        yield self.quick_align()
        
        # Medium pass: Every 10th frame  
        yield self.medium_align()
        
        # Fine pass: Every 3rd frame
        yield self.fine_align()
```

### Phase 4: Algorithm Enhancements (Priority: HIGH)

#### 4.1 Optical Flow Integration
```python
def validate_alignment_with_optical_flow(
    bg_frames: List[np.ndarray],
    fg_frames: List[np.ndarray],
    alignment: TemporalAlignment
) -> TemporalAlignment:
    """
    Use optical flow to verify and correct alignment.
    Detects and fixes drift with sub-frame precision.
    """
    # Calculate optical flow between matched frames
    # Detect anomalies in flow magnitude
    # Adjust alignment where flow indicates mismatch
```

#### 4.2 Content-Aware Matching
```python
def content_aware_temporal_alignment(bg_info, fg_info):
    """
    Adjust matching strategy based on content type.
    """
    # Detect content characteristics:
    # - Static scenes: Sparse matching OK
    # - Fast motion: Dense matching required
    # - Scene changes: Force keyframes at cuts
```

## Implementation Plan

### Quick Wins (1-2 days)
1. Increase minimum keyframe density (change default from 2000 to 200)
2. Enable parallel cost matrix computation
3. Fix border mode to work with perceptual hashing

### Week 1
1. Implement adaptive keyframe density algorithm
2. Add sliding window refinement
3. Optimize perceptual hash computation

### Week 2
1. Implement hierarchical matching
2. Add caching system
3. Create progressive alignment mode

### Week 3
1. Integrate optical flow validation
2. Implement content-aware matching
3. Performance testing and optimization

## Performance Targets

### Current Performance
- 3.5 minutes for 8-second video
- Visible drift with sparse keyframes
- No GPU acceleration

### Target Performance
- < 10 seconds for 8-second video (20x speedup)
- Zero visible drift (< 1 frame deviation)
- Optional GPU acceleration for 50x speedup

### Benchmark Metrics
1. **Alignment Accuracy**: Maximum frame deviation from ground truth
2. **Temporal Stability**: Standard deviation of frame-to-frame timing
3. **Processing Speed**: Frames per second processed
4. **Memory Usage**: Peak memory consumption

## Configuration Changes

### New CLI Options
```bash
--drift-tolerance 16.67  # Max drift in milliseconds
--matching-quality fast|balanced|precise  # Preset configurations
--progressive  # Enable progressive refinement
--cache-dir ./cache  # Enable caching
--gpu  # Enable GPU acceleration
```

### Recommended Defaults
```python
# Change defaults for better out-of-box experience
DEFAULT_MAX_KEYFRAMES = 200  # Was 2000
DEFAULT_DRIFT_TOLERANCE = 16.67  # 1 frame at 60fps
DEFAULT_WINDOW_SIZE = 30  # For sliding window refinement
```

## Testing Strategy

### Test Cases
1. **Uniform Motion**: Videos with constant speed
2. **Variable Motion**: Speed changes throughout
3. **Scene Changes**: Multiple cuts and transitions
4. **Border Visibility**: Various foreground positions
5. **Frame Rate Mismatch**: 24fps to 60fps conversion

### Validation Methods
1. **Visual Inspection**: Frame-by-frame comparison
2. **Drift Measurement**: Plot temporal deviation over time
3. **Performance Profiling**: Identify remaining bottlenecks

## Conclusion

The proposed improvements address both critical issues:
1. **Drift Elimination**: Through dense matching and continuous refinement
2. **Performance**: Through parallelization, hierarchical matching, and GPU acceleration

The phased approach allows incremental improvements while maintaining stability. Quick wins can be implemented immediately for noticeable improvements, while longer-term enhancements provide the full solution.

## Appendix: Detailed Algorithm Specifications

### A.1 Adaptive DTW Window Size
```python
def calculate_dtw_window(fps_ratio: float, confidence: float) -> int:
    """
    Dynamically adjust DTW window based on:
    - FPS ratio between videos
    - Alignment confidence so far
    - Content complexity
    """
    base_window = 100
    fps_factor = abs(1.0 - fps_ratio) * 200
    confidence_factor = (1.0 - confidence) * 50
    return int(base_window + fps_factor + confidence_factor)
```

### A.2 Perceptual Hash Weighting
```python
def weighted_hash_distance(hash1: Dict, hash2: Dict, content_type: str) -> float:
    """
    Weight different hash types based on content.
    """
    weights = {
        'static': {'phash': 0.7, 'color': 0.3},
        'motion': {'phash': 0.5, 'edge': 0.3, 'color': 0.2},
        'transition': {'ahash': 0.8, 'color': 0.2}
    }
    # Apply content-specific weights
```

### A.3 Border Mask Optimization
```python
def create_adaptive_border_mask(spatial_alignment, fg_info, bg_info) -> np.ndarray:
    """
    Create border mask that adapts to content.
    More weight to high-contrast edges.
    """
    # Detect edges in border region
    # Weight mask by edge strength
    # Ensure minimum coverage
```

## Detailed Implementation Guide

### Quick Fix #1: Change Default Max Keyframes

**File**: `src/vidkompy/vidkompy.py`
```python
def main(
    bg: str,
    fg: str,
    output: str | None = None,
    match_time: str = "border",
    match_space: str = "precise",
    temporal_align: str = "dtw",
    skip_spatial_align: bool = False,
    trim: bool = True,
    verbose: bool = False,
    max_keyframes: int = 200,  # CHANGE: Was 2000
    border: int = 8,
    blend: bool = False,
    window: int = 0,
):
```

**Rationale**: With 200 keyframes instead of 2000, you'll sample every ~2.4 frames instead of every ~40 frames. This drastically reduces interpolation gaps and eliminates most drift.

### Quick Fix #2: Parallelize Cost Matrix Building

**File**: `src/vidkompy/core/temporal_alignment.py`

Replace the current `_build_cost_matrix` method with:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

def _build_cost_matrix(
    self,
    bg_info: VideoInfo,
    fg_info: VideoInfo,
    bg_indices: list[int],
    fg_indices: list[int],
) -> np.ndarray | None:
    """Build cost matrix for dynamic programming alignment with parallel processing."""
    n_fg = len(fg_indices)
    n_bg = len(bg_indices)
    
    # Initialize cost matrix
    cost_matrix = np.full((n_fg, n_bg), np.inf)
    
    # Pre-extract all frames to avoid redundant I/O
    logger.info("Pre-extracting frames for cost matrix...")
    resize_factor = 0.25
    
    # Extract frames in parallel batches
    with ThreadPoolExecutor(max_workers=4) as executor:
        fg_future = executor.submit(
            self.processor.extract_frames, fg_info.path, fg_indices, resize_factor
        )
        bg_future = executor.submit(
            self.processor.extract_frames, bg_info.path, bg_indices, resize_factor
        )
        
        fg_frames = fg_future.result()
        bg_frames = bg_future.result()
    
    # Create frame dictionaries
    fg_frame_dict = {idx: frame for idx, frame in zip(fg_indices, fg_frames) if frame is not None}
    bg_frame_dict = {idx: frame for idx, frame in zip(bg_indices, bg_frames) if frame is not None}
    
    # Compute similarities in parallel
    logger.info(f"Computing {n_fg * n_bg} similarities in parallel...")
    
    def compute_cell(i, j):
        """Compute single cell of cost matrix."""
        fg_idx = fg_indices[i]
        bg_idx = bg_indices[j]
        
        if fg_idx not in fg_frame_dict or bg_idx not in bg_frame_dict:
            return i, j, np.inf
            
        fg_frame = fg_frame_dict[fg_idx]
        bg_frame = bg_frame_dict[bg_idx]
        
        # Apply mask if in border mode
        if self._current_mask is not None:
            similarity = self._compute_frame_similarity(bg_frame, fg_frame, self._current_mask)
        else:
            similarity = self._compute_frame_similarity(bg_frame, fg_frame)
        
        cost = 1.0 - similarity
        
        # Add temporal consistency penalty
        expected_j = int(i * n_bg / n_fg)
        time_penalty = 0.1 * abs(j - expected_j) / n_bg
        cost += time_penalty
        
        return i, j, cost
    
    # Process in parallel
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("  Computing similarities...", total=n_fg * n_bg)
        
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            # Submit all tasks
            futures = []
            for i in range(n_fg):
                for j in range(n_bg):
                    future = executor.submit(compute_cell, i, j)
                    futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                i, j, cost = future.result()
                cost_matrix[i, j] = cost
                progress.update(task, advance=1)
    
    return cost_matrix
```

### Quick Fix #3: Enable Masked Perceptual Hashing

**File**: `src/vidkompy/core/frame_fingerprint.py`

Add this method to the `FrameFingerprinter` class:

```python
def compute_masked_fingerprint(
    self, frame: np.ndarray, mask: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute fingerprint for masked region of frame.
    
    Args:
        frame: Input frame
        mask: Binary mask (1 = include, 0 = exclude)
        
    Returns:
        Fingerprint dictionary
    """
    # Apply mask to frame
    masked_frame = frame.copy()
    if len(frame.shape) == 3:
        # Apply to all channels
        for c in range(frame.shape[2]):
            masked_frame[:, :, c] = frame[:, :, c] * mask
    else:
        masked_frame = frame * mask
    
    # Crop to bounding box of mask to focus on relevant region
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    cropped = masked_frame[rmin:rmax+1, cmin:cmax+1]
    
    # Compute fingerprint on cropped region
    return self.compute_fingerprint(cropped)
```

Then modify `temporal_alignment.py` to use masked fingerprints in border mode:

```python
# In _align_frames_dtw method, after computing fingerprints:
if self._current_mask is not None:
    logger.info("Computing masked fingerprints for border mode...")
    # Apply mask to fingerprints
    masked_fg_fingerprints = {}
    masked_bg_fingerprints = {}
    
    for idx in fg_fingerprints:
        frame = self.processor.extract_frames(fg_info.path, [idx], 0.25)[0]
        if frame is not None:
            masked_fg_fingerprints[idx] = self.fingerprinter.compute_masked_fingerprint(
                frame, self._current_mask
            )
    
    for idx in bg_fingerprints:
        frame = self.processor.extract_frames(bg_info.path, [idx], 0.25)[0]
        if frame is not None:
            masked_bg_fingerprints[idx] = self.fingerprinter.compute_masked_fingerprint(
                frame, self._current_mask
            )
    
    fg_fingerprints = masked_fg_fingerprints
    bg_fingerprints = masked_bg_fingerprints
```

### Medium-Term Improvements

#### 1. Adaptive Keyframe Density

**File**: `src/vidkompy/core/temporal_alignment.py`

Add this method:

```python
def calculate_adaptive_keyframe_count(
    self, 
    fg_info: VideoInfo, 
    bg_info: VideoInfo,
    target_drift_frames: float = 1.0
) -> int:
    """Calculate optimal keyframe count to prevent drift.
    
    Args:
        fg_info: Foreground video info
        bg_info: Background video info  
        target_drift_frames: Maximum acceptable drift in frames
        
    Returns:
        Optimal number of keyframes
    """
    # Account for FPS difference
    fps_ratio = abs(bg_info.fps - fg_info.fps) / max(bg_info.fps, fg_info.fps)
    
    # More keyframes needed for higher FPS mismatch
    fps_factor = 1.0 + fps_ratio * 2.0
    
    # Calculate base requirement
    base_keyframes = fg_info.frame_count / (target_drift_frames * 10)
    
    # Apply factors
    required_keyframes = int(base_keyframes * fps_factor)
    
    # Clamp to reasonable range
    return max(50, min(required_keyframes, fg_info.frame_count // 2))
```

Then use it in `_find_keyframe_matches`:

```python
# Replace fixed sampling with adaptive
adaptive_keyframes = self.calculate_adaptive_keyframe_count(fg_info, bg_info)
effective_target_keyframes = min(self.max_keyframes, adaptive_keyframes)
logger.info(f"Using {effective_target_keyframes} keyframes (adaptive calculation)")
```

#### 2. Sliding Window Refinement

Add this after initial keyframe matching:

```python
def refine_with_sliding_window(
    self,
    initial_matches: list[tuple[int, int, float]],
    bg_info: VideoInfo,
    fg_info: VideoInfo,
    window_size: int = 20,
    threshold: float = 0.7
) -> list[tuple[int, int, float]]:
    """Refine alignment by adding intermediate matches where needed."""
    refined = []
    
    for i in range(len(initial_matches) - 1):
        curr = initial_matches[i]
        next = initial_matches[i + 1]
        
        refined.append(curr)
        
        # Check if gap is too large
        fg_gap = next[1] - curr[1]
        if fg_gap > window_size:
            # Sample intermediate points
            n_intermediate = fg_gap // window_size
            
            for j in range(1, n_intermediate + 1):
                # Interpolate position
                alpha = j / (n_intermediate + 1)
                fg_intermediate = int(curr[1] + alpha * (next[1] - curr[1]))
                bg_intermediate = int(curr[0] + alpha * (next[0] - curr[0]))
                
                # Verify with actual matching in small window
                best_match = self._find_best_match_in_window(
                    fg_intermediate, bg_intermediate, 
                    window_size // 2, bg_info, fg_info
                )
                
                if best_match[2] > threshold:
                    refined.append(best_match)
    
    refined.append(initial_matches[-1])
    return refined
```

### Performance Benchmarking

Create a benchmarking script:

```python
#!/usr/bin/env python3
# benchmark.py

import time
import numpy as np
from pathlib import Path
from vidkompy import main

def benchmark_alignment(bg_path: str, fg_path: str, config: dict) -> dict:
    """Benchmark video alignment with given configuration."""
    start_time = time.time()
    
    output_path = f"benchmark_{config['name']}.mp4"
    
    main(
        bg=bg_path,
        fg=fg_path,
        output=output_path,
        **config['params']
    )
    
    elapsed = time.time() - start_time
    
    # Analyze drift (requires ground truth or visual inspection)
    # For now, just return timing
    return {
        'config': config['name'],
        'time': elapsed,
        'fps': 483 / elapsed  # frames processed per second
    }

# Test configurations
configs = [
    {
        'name': 'sparse_keyframes',
        'params': {'max_keyframes': 12}
    },
    {
        'name': 'dense_keyframes', 
        'params': {'max_keyframes': 200}
    },
    {
        'name': 'adaptive_dense',
        'params': {'max_keyframes': 100, 'window': 20}
    },
    {
        'name': 'dtw_perceptual',
        'params': {'temporal_align': 'dtw', 'max_keyframes': 100}
    }
]

# Run benchmarks
results = []
for config in configs:
    print(f"Running benchmark: {config['name']}")
    result = benchmark_alignment('tests/bg.mp4', 'tests/fg.mp4', config)
    results.append(result)
    print(f"  Time: {result['time']:.2f}s, FPS: {result['fps']:.2f}")

# Summary
print("\nBenchmark Summary:")
print("-" * 40)
for r in results:
    print(f"{r['config']:20s} {r['time']:6.2f}s  {r['fps']:6.2f} fps")
```

## Conclusion

The improvements outlined in this specification will transform vidkompy from a slow, drift-prone tool into a fast, accurate video alignment system. The phased approach allows for incremental improvements while maintaining stability, with quick wins available immediately and comprehensive solutions following in subsequent phases.
