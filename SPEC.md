# Implementation Plan: Tunnel-Based Temporal Alignment

## Overview

The user has proposed a new approach to temporal alignment that avoids perceptual hashing/fingerprinting and instead uses direct frame comparison with sliding windows. This approach will be implemented as two new temporal alignment engines:

1. **tunnel_full**: Uses full frame comparison
2. **tunnel_mask**: Uses masked frame comparison (focusing on the overlapping region)

## Key Concept

Instead of pre-computing fingerprints for all frames, this approach:
- Iterates through each foreground (FG) frame sequentially
- Uses a sliding window in the background (BG) video to find the best match
- Performs matching from both ends (start and end) of the video
- Enforces monotonicity by constraining the search window to frames after the previous match

## Implementation Details

### 1. Tunnel Full Engine (`tunnel_full`)

#### Algorithm:
1. **Forward Pass (from start)**:
   - Start with FG frame 0
   - Search within a window of BG frames (e.g., frames 0-30) for the best match
   - Best match = minimum pixel difference between FG frame and BG frame (cropped to FG region)
   - Once found, move to FG frame 1
   - Constrain BG search window to start from the matched BG frame index + 1
   - Continue until all FG frames are matched

2. **Backward Pass (from end)**:
   - Start with last FG frame
   - Search within a window of BG frames near the end for the best match
   - Once found, move to previous FG frame
   - Constrain BG search window to end before the matched BG frame index - 1
   - Continue backward until all FG frames are matched

3. **Merge Strategy**:
   - Combine forward and backward mappings
   - Use confidence scores based on match quality
   - Apply smoothing to eliminate any remaining discontinuities

#### Frame Comparison Method:
```python
def compute_frame_difference(fg_frame, bg_frame, x_offset, y_offset):
    # Crop BG frame to FG region
    bg_cropped = bg_frame[y_offset:y_offset+fg_height, x_offset:x_offset+fg_width]
    
    # Compute pixel-wise difference
    diff = np.abs(fg_frame.astype(float) - bg_cropped.astype(float))
    
    # Return mean absolute difference
    return np.mean(diff)
```

### 2. Tunnel Mask Engine (`tunnel_mask`)

Similar to `tunnel_full` but with an additional masking step:

#### Masking Strategy:
1. Create a binary mask identifying the actual content region in FG frames (non-black areas)
2. Apply this mask during frame comparison to focus only on content regions
3. This helps when FG video has letterboxing or pillarboxing

#### Modified Frame Comparison:
```python
def compute_masked_frame_difference(fg_frame, bg_frame, x_offset, y_offset, mask):
    # Crop BG frame to FG region
    bg_cropped = bg_frame[y_offset:y_offset+fg_height, x_offset:x_offset+fg_width]
    
    # Apply mask to both frames
    fg_masked = fg_frame * mask
    bg_masked = bg_cropped * mask
    
    # Compute pixel-wise difference only in masked region
    diff = np.abs(fg_masked.astype(float) - bg_masked.astype(float))
    
    # Return mean absolute difference normalized by mask area
    return np.sum(diff) / np.sum(mask)
```

## Implementation Steps

### Step 1: Create Base Tunnel Aligner Class
Create `src/vidkompy/core/tunnel_aligner.py`:
- Base class with common functionality
- Window management logic
- Monotonicity enforcement
- Forward/backward pass framework

### Step 2: Implement TunnelFullAligner
Create subclass in same file:
- Implement `compute_frame_difference` method
- Handle full frame comparison logic
- Optimize with early stopping when good match found

### Step 3: Implement TunnelMaskAligner
Create subclass in same file:
- Implement mask generation logic
- Implement `compute_masked_frame_difference` method
- Handle edge cases where mask might be empty

### Step 4: Integration
Update `src/vidkompy/core/alignment_engine.py`:
- Add `tunnel_full` and `tunnel_mask` to engine registry
- Create appropriate configuration classes
- Wire up CLI parameters

### Step 5: Optimization Considerations
1. **Frame Reading**: Use sequential reading where possible
2. **Downsampling**: Option to downsample frames for faster comparison
3. **Early Stopping**: Stop searching when difference is below threshold
4. **Parallel Processing**: Process forward and backward passes in parallel
5. **Caching**: Cache recently read frames to avoid re-reading

## Configuration Parameters

### Common Parameters:
- `window_size`: Size of the sliding window (default: 30 frames)
- `downsample_factor`: Factor to downsample frames (default: 1.0 = no downsampling)
- `early_stop_threshold`: Stop searching if difference below this (default: 0.05)
- `merge_strategy`: How to combine forward/backward passes ("average", "confidence_weighted")

### Tunnel Mask Specific:
- `mask_threshold`: Threshold for creating binary mask (default: 10/255)
- `mask_erosion`: Pixels to erode mask to avoid edge artifacts (default: 2)

## Expected Benefits

1. **Direct Comparison**: No information loss from fingerprinting
2. **Adaptive**: Can find matches even with compression artifacts
3. **Monotonic by Design**: The sliding window constraint ensures monotonicity
4. **Bidirectional**: Matching from both ends improves robustness
5. **Content-Aware**: Mask mode focuses on actual content, ignoring borders

## Potential Challenges

1. **Performance**: Direct pixel comparison is slower than fingerprint matching
2. **Memory**: May need to keep more frames in memory
3. **Noise Sensitivity**: Pixel-wise comparison sensitive to compression/noise
4. **Window Size**: Too small = might miss correct match; too large = slow

## Mitigation Strategies

1. Use downsampling for initial coarse matching, then refine
2. Implement efficient frame caching
3. Add noise-robust comparison metrics (e.g., SSIM as alternative)
4. Make window size adaptive based on initial probe