# SPEC4: Comprehensive vidkompy Performance and Matching Improvement Plan

## 1. Executive Summary

The current vidkompy implementation suffers from two critical issues:
1. **Performance**: Processing an 8-second video takes 5+ minutes (should be <10 seconds)
2. **Matching Quality**: Finding 0 monotonic keyframe matches indicates algorithmic failure

This specification provides a detailed plan to achieve:
- **10-50x performance improvement** through perceptual hashing and parallelization
- **Correct monotonic frame mapping** via Dynamic Time Warping (DTW)
- **Robust matching** with multi-scale validation

## 2. Current State Analysis

### 2.1. Performance Bottlenecks

From the log analysis:
```
20:41:45 → 20:48:48 = 7 minutes 3 seconds for 483 frames
```

Breakdown:
1. **Frame extraction**: ~2-3s per frame pair (SSIM computation on full resolution)
2. **Cost matrix building**: 483 × 472 = 228,276 comparisons
3. **No parallelization**: Single-threaded processing
4. **Redundant computation**: Re-extracting frames multiple times

### 2.2. Matching Algorithm Issues

1. **Independent matching**: Each fg frame finds "best" bg match without temporal context
2. **Broken monotonic filtering**: Removes most matches, leaving only 5 keyframes
3. **Poor interpolation**: Extreme stretching/compression between sparse keyframes
4. **No continuity enforcement**: Allows backward jumps and extreme forward leaps

## 3. Proposed Solution Architecture

### 3.1. Fast Frame Fingerprinting System

#### 3.1.1. Perceptual Hashing Implementation
```python
class FrameFingerprinter:
    """Ultra-fast frame comparison using perceptual hashing.
    
    Why: Perceptual hashes are 100-1000x faster than SSIM while maintaining
    good accuracy for similar frames. They're robust to compression artifacts
    and minor color/brightness changes.
    """
    
    def __init__(self):
        # Multiple hash algorithms for robustness
        self.phash = cv2.img_hash.PHash_create()
        self.ahash = cv2.img_hash.AverageHash_create()
        self.dhash = cv2.img_hash.ColorMomentHash_create()
        
        # Cache for computed hashes
        self.hash_cache = {}
        
    def compute_fingerprint(self, frame: np.ndarray) -> dict:
        """Compute multi-algorithm fingerprint.
        
        Why multiple algorithms: Different hashes capture different aspects
        - pHash: Frequency domain, good for structure
        - aHash: Average color, good for brightness
        - dHash: Gradients, good for edges
        """
        # Resize to standard size for consistent hashing
        std_frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale for hash computation
        gray = cv2.cvtColor(std_frame, cv2.COLOR_BGR2GRAY)
        
        fingerprint = {
            'phash': self.phash.compute(gray),
            'ahash': self.ahash.compute(gray),
            'dhash': self.dhash.compute(gray),
            'histogram': self._compute_color_histogram(std_frame)
        }
        
        return fingerprint
    
    def compare_fingerprints(self, fp1: dict, fp2: dict) -> float:
        """Compare fingerprints with weighted scoring.
        
        Why weighted: pHash is most reliable for video frames,
        but other hashes help disambiguate similar frames.
        """
        scores = {
            'phash': 1.0 - (cv2.norm(fp1['phash'], fp2['phash'], cv2.NORM_HAMMING) / 64),
            'ahash': 1.0 - (cv2.norm(fp1['ahash'], fp2['ahash'], cv2.NORM_HAMMING) / 64),
            'dhash': 1.0 - (cv2.norm(fp1['dhash'], fp2['dhash'], cv2.NORM_HAMMING) / 64),
            'histogram': cv2.compareHist(fp1['histogram'], fp2['histogram'], cv2.HISTCMP_CORREL)
        }
        
        # Weighted combination
        weights = {'phash': 0.5, 'ahash': 0.2, 'dhash': 0.2, 'histogram': 0.1}
        total_score = sum(scores[k] * weights[k] for k in scores)
        
        return total_score
```

#### 3.1.2. Parallel Hash Computation
```python
def precompute_all_fingerprints(self, video_path: str, indices: List[int]) -> dict:
    """Compute all fingerprints in parallel.
    
    Why parallel: Frame extraction is I/O bound, hash computation is CPU bound.
    Using process pool for CPU work and thread pool for I/O gives best performance.
    """
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    
    # Step 1: Extract frames in parallel (I/O bound)
    with ThreadPoolExecutor(max_workers=8) as executor:
        frame_futures = {
            idx: executor.submit(self._extract_single_frame, video_path, idx)
            for idx in indices
        }
        
        frames = {}
        for idx, future in frame_futures.items():
            frame = future.result()
            if frame is not None:
                frames[idx] = frame
    
    # Step 2: Compute hashes in parallel (CPU bound)
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        hash_futures = {
            idx: executor.submit(self.compute_fingerprint, frame)
            for idx, frame in frames.items()
        }
        
        fingerprints = {}
        for idx, future in hash_futures.items():
            fingerprints[idx] = future.result()
    
    return fingerprints
```

### 3.2. Dynamic Time Warping (DTW) for Monotonic Alignment

#### 3.2.1. Why DTW?
- **Guarantees monotonicity**: No backward jumps possible
- **Global optimization**: Finds best overall path, not greedy local matches
- **Handles speed variations**: Can map 1 fg frame to multiple bg frames or vice versa
- **Proven algorithm**: Standard in speech recognition and time series analysis

#### 3.2.2. Optimized DTW Implementation
```python
class DTWAligner:
    """Dynamic Time Warping for video frame alignment.
    
    Why DTW: Unlike the current greedy matching, DTW finds the globally
    optimal monotonic path through the similarity matrix. This prevents
    the chaotic jumping seen in current implementation.
    """
    
    def __init__(self, window_constraint: int = 100):
        """Initialize with Sakoe-Chiba band constraint.
        
        Why window constraint: Reduces O(N²) to O(N×window), making
        DTW practical for long videos while preventing extreme warping.
        """
        self.window = window_constraint
        
    def align_videos(
        self, 
        fg_fingerprints: dict,
        bg_fingerprints: dict,
        fingerprint_compare_func
    ) -> List[Tuple[int, int, float]]:
        """Find optimal monotonic alignment using DTW.
        
        Returns list of (fg_idx, bg_idx, confidence) tuples.
        """
        fg_indices = sorted(fg_fingerprints.keys())
        bg_indices = sorted(bg_fingerprints.keys())
        
        n_fg = len(fg_indices)
        n_bg = len(bg_indices)
        
        # Initialize DTW matrix with infinity
        dtw = np.full((n_fg + 1, n_bg + 1), np.inf)
        dtw[0, 0] = 0
        
        # Cost matrix (distance, not similarity)
        for i in range(1, n_fg + 1):
            # Sakoe-Chiba band constraint
            j_start = max(1, i - self.window)
            j_end = min(n_bg + 1, i + self.window)
            
            for j in range(j_start, j_end):
                cost = 1.0 - fingerprint_compare_func(
                    fg_fingerprints[fg_indices[i-1]],
                    bg_fingerprints[bg_indices[j-1]]
                )
                
                # DTW recursion: min of three possible paths
                dtw[i, j] = cost + min(
                    dtw[i-1, j],      # insertion (skip bg frame)
                    dtw[i, j-1],      # deletion (skip fg frame)
                    dtw[i-1, j-1]     # match
                )
        
        # Backtrack to find optimal path
        path = self._backtrack_path(dtw, n_fg, n_bg)
        
        # Convert to frame alignments
        alignments = []
        for i, j in path:
            if i > 0 and j > 0:  # Skip dummy start position
                confidence = fingerprint_compare_func(
                    fg_fingerprints[fg_indices[i-1]],
                    bg_fingerprints[bg_indices[j-1]]
                )
                alignments.append((
                    fg_indices[i-1],
                    bg_indices[j-1],
                    confidence
                ))
        
        return alignments
    
    def _backtrack_path(self, dtw: np.ndarray, i: int, j: int) -> List[Tuple[int, int]]:
        """Backtrack through DTW matrix to find optimal path."""
        path = [(i, j)]
        
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                # Choose path with minimum cost
                candidates = [
                    (i-1, j, dtw[i-1, j]),
                    (i, j-1, dtw[i, j-1]),
                    (i-1, j-1, dtw[i-1, j-1])
                ]
                i, j, _ = min(candidates, key=lambda x: x[2])
            
            path.append((i, j))
        
        return list(reversed(path))
```

### 3.3. Multi-Scale Validation Pipeline

```python
class MultiScaleValidator:
    """Validates matches at multiple resolutions for robustness.
    
    Why multi-scale: Quick rejection of bad matches at low resolution,
    accurate validation only for promising matches at high resolution.
    """
    
    def __init__(self):
        self.scales = [0.125, 0.25, 0.5, 1.0]  # Coarse to fine
        self.thresholds = [0.5, 0.6, 0.7, 0.8]  # Increasing strictness
        
    def validate_match(
        self, 
        fg_frame: np.ndarray,
        bg_frame: np.ndarray,
        spatial_offset: Tuple[int, int]
    ) -> Tuple[bool, float]:
        """Validate match with early rejection at coarse scales."""
        
        for scale, threshold in zip(self.scales, self.thresholds):
            # Resize frames
            fg_scaled = cv2.resize(
                fg_frame, 
                (int(fg_frame.shape[1] * scale), int(fg_frame.shape[0] * scale))
            )
            bg_scaled = cv2.resize(
                bg_frame,
                (int(bg_frame.shape[1] * scale), int(bg_frame.shape[0] * scale))
            )
            
            # Apply spatial offset at this scale
            x_off = int(spatial_offset[0] * scale)
            y_off = int(spatial_offset[1] * scale)
            
            # Extract ROI from background
            bg_roi = self._extract_roi(bg_scaled, fg_scaled.shape, (x_off, y_off))
            
            # Compute SSIM
            ssim_score = structural_similarity(
                cv2.cvtColor(fg_scaled, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY)
            )
            
            # Early rejection
            if ssim_score < threshold:
                return False, ssim_score
        
        # Passed all scales
        return True, ssim_score
```

### 3.4. Optimized Frame Mapping

```python
def create_optimal_frame_mapping(
    keyframe_alignments: List[Tuple[int, int, float]],
    total_fg_frames: int,
    total_bg_frames: int,
    target_fps: float
) -> Dict[int, int]:
    """Create frame mapping that preserves fg timing.
    
    Why this approach: The fg video is the reference (better quality),
    so we preserve its timing exactly and find optimal bg frames.
    """
    
    # Ensure monotonic keyframes
    keyframe_alignments.sort(key=lambda x: x[0])
    
    # Add anchors if missing
    if keyframe_alignments[0][0] != 0:
        keyframe_alignments.insert(0, (0, 0, 1.0))
    
    if keyframe_alignments[-1][0] != total_fg_frames - 1:
        # Proportional final mapping
        last_fg = total_fg_frames - 1
        last_bg = min(
            total_bg_frames - 1,
            keyframe_alignments[-1][1] + 
            int((last_fg - keyframe_alignments[-1][0]) * 
                (total_bg_frames / total_fg_frames))
        )
        keyframe_alignments.append((last_fg, last_bg, 0.8))
    
    # Build mapping with smooth interpolation
    mapping = {}
    
    for i in range(len(keyframe_alignments) - 1):
        fg_start, bg_start, _ = keyframe_alignments[i]
        fg_end, bg_end, _ = keyframe_alignments[i + 1]
        
        # Use cubic interpolation for smoother motion
        fg_range = np.arange(fg_start, fg_end + 1)
        bg_range = np.interp(
            fg_range,
            [fg_start, fg_end],
            [bg_start, bg_end]
        )
        
        for fg_idx, bg_idx in zip(fg_range, bg_range):
            mapping[int(fg_idx)] = int(round(bg_idx))
    
    return mapping
```

## 4. Implementation Roadmap

### 4.1. Phase 1: Performance Foundation (1-2 days)
1. **Install opencv-contrib-python** for img_hash module
2. **Implement FrameFingerprinter** class with caching
3. **Add parallel fingerprint computation**
4. **Benchmark**: Target 100+ fps for fingerprinting

### 4.2. Phase 2: DTW Integration (2-3 days)
1. **Implement DTWAligner** with Sakoe-Chiba constraint
2. **Replace current keyframe matching logic**
3. **Add progress reporting for DTW computation**
4. **Test**: Verify monotonic output on test videos

### 4.3. Phase 3: Validation & Refinement (1-2 days)
1. **Implement MultiScaleValidator**
2. **Add confidence-based keyframe selection**
3. **Implement smooth frame interpolation**
4. **Optimize memory usage with frame caching**

### 4.4. Phase 4: Production Optimization (1 day)
1. **Add CLI option**: `--temporal_align dtw`
2. **Profile and optimize hot paths**
3. **Add detailed logging for debugging**
4. **Update documentation**

## 5. Performance Targets

### 5.1. Current Performance
- 8-second video: 420+ seconds (7+ minutes)
- Frame matching: ~2 fps
- Memory usage: Unknown but likely high

### 5.2. Target Performance
- 8-second video: <10 seconds
- Frame matching: 50+ fps with fingerprinting
- Memory usage: <500MB for 1080p videos
- Quality: Zero temporal drift, smooth motion

## 6. Testing Strategy

### 6.1. Unit Tests
1. **Fingerprinting accuracy**: Known similar/different frames
2. **DTW correctness**: Synthetic offset recovery
3. **Interpolation smoothness**: No frame jumps

### 6.2. Integration Tests
1. **Test videos**: bg.mp4 + fg.mp4 with known good alignment
2. **Performance benchmarks**: Track regression
3. **Quality metrics**: SSIM scores for aligned frames

### 6.3. Visual Validation
1. **Motion smoothness**: No speedup/slowdown artifacts
2. **Alignment accuracy**: Objects properly overlapped
3. **Temporal consistency**: Action progresses naturally

## 7. Risk Mitigation

### 7.1. Potential Issues
1. **Hash collisions**: Multiple algorithms reduce false matches
2. **Memory usage**: Streaming processing for long videos
3. **DTW complexity**: Sakoe-Chiba band keeps it O(N)
4. **Quality degradation**: Multi-scale validation ensures accuracy

### 7.2. Fallback Strategy
Keep current SSIM-based matching as fallback option accessible via `--temporal_align classic`

## 8. Success Metrics

1. **Performance**: 10-50x speedup achieved
2. **Quality**: No visible temporal artifacts
3. **Robustness**: Handles various video formats/resolutions
4. **Usability**: Clear progress indication, helpful error messages

## 9. Conclusion

This plan addresses both performance and quality issues through:
- **Perceptual hashing**: Orders of magnitude faster comparison
- **DTW algorithm**: Guaranteed monotonic, optimal alignment
- **Parallel processing**: Full CPU utilization
- **Smart validation**: Fast rejection, accurate confirmation

The implementation is incremental with clear milestones and fallback options, ensuring we can deliver improvements without breaking existing functionality.