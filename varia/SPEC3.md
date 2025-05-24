# SPEC3: Detailed Analysis and Solution Plan for vidkompy Frame Matching Issues

## Problem Analysis

The current implementation has critical flaws in its temporal alignment algorithm that result in:

1. **Non-monotonic frame matching**: The log shows chaotic jumping (bg[250]→fg[1], bg[250]→fg[2], bg[0]→fg[3]), violating the fundamental constraint that video action is continuous
2. **Extreme speed distortion**: The fg video compresses its entire action into ~1 second, then becomes static for the remaining 7 seconds
3. **Poor keyframe selection**: The algorithm matches frames without considering temporal continuity
4. **Incorrect filtering logic**: The monotonic filter removes good matches while keeping bad ones

## Root Cause Analysis

### 1. Flawed Similarity Matching
The current code finds the "best" bg match for each fg frame independently, without considering temporal context. This leads to:
- Frame bg[250] being selected for multiple early fg frames
- Frame bg[448] being selected for many middle fg frames
- No enforcement of forward progression

### 2. Broken Monotonic Filtering
The `_filter_monotonic` function filters out matches where bg indices don't increase monotonically, but:
- It doesn't ensure the remaining matches span the video properly
- After filtering, only 5 matches remain from 483 fg frames
- These 5 matches cause extreme interpolation errors

### 3. Poor Interpolation Strategy
With only 5 keyframe matches, the interpolation must map:
- Frames 0-100 of fg to a tiny range in bg (causing speedup)
- Frames 100-483 of fg to nearly the same bg frame (causing static output)

## Proposed Solution

### Phase 1: Implement Proper Monotonic Constraint Enforcement

Instead of finding best matches then filtering, we need to:

1. **Use Dynamic Programming for optimal monotonic path**:
   - Build a cost matrix of frame similarities
   - Find the optimal path that minimizes total dissimilarity while maintaining monotonicity
   - This guarantees every fg frame maps to a sensible bg frame

2. **Implement constrained window search**:
   - For each fg frame i, only search bg frames in range [last_matched_bg, last_matched_bg + max_advance]
   - This prevents backward jumps and limits forward jumps

### Phase 2: Smart Keyframe Selection

1. **Uniform temporal sampling**: Sample keyframes evenly across the video duration
2. **Quality-based refinement**: Add additional keyframes where similarity is low
3. **Boundary anchoring**: Always include first and last frames as keyframes

### Phase 3: Improved Similarity Metrics

As suggested in SPEC2, implement:
1. **Perceptual hashing** for fast coarse matching
2. **Downsampled SSIM** for accurate matching on reduced resolution
3. **Multi-metric validation** to ensure matches are correct

## Detailed Implementation Plan

### 1. Fix Temporal Alignment Core Algorithm

```python
def align_frames_monotonic(self, fg_frames, bg_frames, max_frames=2000):
    """
    Implement proper monotonic frame alignment using dynamic programming.
    """
    # Step 1: Sample frames uniformly
    fg_indices = np.linspace(0, len(fg_frames)-1, min(max_frames, len(fg_frames)), dtype=int)
    
    # Step 2: Compute similarity matrix with perceptual hashes
    fg_hashes = [self.compute_phash(fg_frames[i]) for i in fg_indices]
    bg_hashes = [self.compute_phash(bg_frames[i]) for i in range(len(bg_frames))]
    
    # Step 3: Build cost matrix (lower is better)
    cost_matrix = np.zeros((len(fg_indices), len(bg_frames)))
    for i, fg_idx in enumerate(fg_indices):
        for j in range(len(bg_frames)):
            # Hamming distance between perceptual hashes
            cost_matrix[i, j] = self.hamming_distance(fg_hashes[i], bg_hashes[j])
    
    # Step 4: Dynamic programming to find optimal monotonic path
    dp = np.full_like(cost_matrix, np.inf)
    path = np.zeros_like(cost_matrix, dtype=int)
    
    # Initialize first row
    dp[0, :] = cost_matrix[0, :]
    
    # Fill DP table
    for i in range(1, len(fg_indices)):
        for j in range(len(bg_frames)):
            # Can only come from previous bg frames
            for k in range(j + 1):
                if dp[i-1, k] + cost_matrix[i, j] < dp[i, j]:
                    dp[i, j] = dp[i-1, k] + cost_matrix[i, j]
                    path[i, j] = k
    
    # Step 5: Backtrack to find optimal path
    matches = []
    j = np.argmin(dp[-1, :])  # Best ending position
    
    for i in range(len(fg_indices) - 1, -1, -1):
        matches.append((fg_indices[i], j))
        if i > 0:
            j = path[i, j]
    
    matches.reverse()
    
    # Step 6: Validate with SSIM on downsampled frames
    validated_matches = []
    for fg_idx, bg_idx in matches:
        # Quick SSIM check on 1/4 resolution
        fg_small = cv2.resize(fg_frames[fg_idx], (480, 270))
        bg_small = cv2.resize(bg_frames[bg_idx], (480, 270))
        ssim_score = self.compute_ssim(fg_small, bg_small)
        
        if ssim_score > 0.5:  # Lower threshold for downsampled
            validated_matches.append({
                'fg_idx': fg_idx,
                'bg_idx': bg_idx,
                'confidence': ssim_score
            })
    
    return validated_matches
```

### 2. Implement Constrained Window Search

```python
def find_best_match_constrained(self, fg_frame, bg_frames, last_bg_idx, window_size=60):
    """
    Find best match within a forward-looking window only.
    """
    start_idx = last_bg_idx
    end_idx = min(last_bg_idx + window_size, len(bg_frames))
    
    best_score = -1
    best_idx = start_idx
    
    for bg_idx in range(start_idx, end_idx):
        score = self.compare_frames_fast(fg_frame, bg_frames[bg_idx])
        if score > best_score:
            best_score = score
            best_idx = bg_idx
    
    return best_idx, best_score
```

### 3. Add Perceptual Hashing

```python
def setup_hashing(self):
    """Initialize perceptual hashing for fast frame comparison."""
    self.hasher = cv2.img_hash.PHash_create()
    
def compute_phash(self, frame):
    """Compute perceptual hash of a frame."""
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Resize to standard size for hashing
    resized = cv2.resize(gray, (32, 32))
    
    # Compute hash
    hash_val = self.hasher.compute(resized)
    return hash_val

def hamming_distance(self, hash1, hash2):
    """Compute Hamming distance between two hashes."""
    return cv2.img_hash.PHash.compare(hash1, hash2)
```

### 4. Improve Interpolation

```python
def interpolate_frame_mapping(self, keyframe_matches, total_fg_frames, total_bg_frames):
    """
    Create smooth frame mapping ensuring temporal consistency.
    """
    # Ensure we have start and end anchors
    if keyframe_matches[0]['fg_idx'] != 0:
        keyframe_matches.insert(0, {'fg_idx': 0, 'bg_idx': 0})
    if keyframe_matches[-1]['fg_idx'] != total_fg_frames - 1:
        # Proportional mapping for last frame
        last_bg = int(keyframe_matches[-1]['bg_idx'] + 
                     (total_fg_frames - 1 - keyframe_matches[-1]['fg_idx']) * 
                     (total_bg_frames / total_fg_frames))
        last_bg = min(last_bg, total_bg_frames - 1)
        keyframe_matches.append({'fg_idx': total_fg_frames - 1, 'bg_idx': last_bg})
    
    # Create mapping for all frames
    frame_map = {}
    
    for i in range(len(keyframe_matches) - 1):
        start_fg = keyframe_matches[i]['fg_idx']
        end_fg = keyframe_matches[i + 1]['fg_idx']
        start_bg = keyframe_matches[i]['bg_idx']
        end_bg = keyframe_matches[i + 1]['bg_idx']
        
        # Linear interpolation between keyframes
        for fg_idx in range(start_fg, end_fg + 1):
            if end_fg == start_fg:
                progress = 0
            else:
                progress = (fg_idx - start_fg) / (end_fg - start_fg)
            
            bg_idx = start_bg + progress * (end_bg - start_bg)
            frame_map[fg_idx] = int(round(bg_idx))
    
    return frame_map
```

## Testing Strategy

1. **Synthetic test**: Create a known temporal offset and verify recovery
2. **Visual inspection**: Check that action progresses smoothly in output
3. **Metrics validation**: Ensure SSIM/hash scores are reasonable for matched frames
4. **Performance testing**: Verify processing completes in reasonable time

## Implementation Priority

1. **Immediate fix**: Replace current keyframe matching with monotonic DP algorithm
2. **Quick win**: Add perceptual hashing for 10-50x speedup
3. **Quality improvement**: Implement constrained window search
4. **Polish**: Add multi-resolution SSIM validation

## Expected Outcomes

1. **Correct temporal alignment**: Action will progress naturally without speedup/static sections
2. **Improved performance**: 10-50x faster frame matching with hashing
3. **Robust matching**: Better handling of compression artifacts and slight differences
4. **Predictable behavior**: Monotonic constraint ensures sensible output

This plan addresses all the issues identified in TODO.md while incorporating the performance improvements suggested in SPEC2.md. The implementation is incremental and testable at each stage.