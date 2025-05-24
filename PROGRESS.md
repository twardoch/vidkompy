# Progress: vidkompy Performance Improvements

## Previous Rounds Complete

### Phase 1-6: Core Architecture ✓
- [x] Modular architecture implemented
- [x] Spatial alignment (template/feature) working
- [x] Temporal alignment (audio/frames) working
- [x] Frame preservation guaranteed
- [x] Basic functionality verified

## Current Round: Performance Optimization (from SPEC2.md)

### Phase 7: Fast Frame Similarity Metrics

- [ ] Install opencv-contrib-python for img_hash module
- [ ] Implement perceptual hashing (pHash/dHash) for frame fingerprinting
- [ ] Pre-compute hashes for all frames at video load
- [ ] Create hash-based similarity function replacing SSIM
- [ ] Use Hamming distance for fast similarity checks
- [ ] Keep SSIM as fallback for low-confidence matches
- [ ] Benchmark hash vs SSIM performance

### Phase 8: Optimize Keyframe Matching

- [ ] Implement adaptive search windows based on previous matches
- [ ] Add confidence-based early termination
- [ ] Reduce initial downsampling from 0.25 to 0.125 for speed
- [ ] Cache extracted frames to avoid redundant reads
- [ ] Implement dynamic keyframe sampling (coarse → fine)
- [ ] Add temporal consistency checks

### Phase 9: Advanced Temporal Alignment (DTW)

- [ ] Implement Dynamic Time Warping algorithm
- [ ] Create cost matrix using perceptual hash distances
- [ ] Add Sakoe-Chiba band constraint for O(N) complexity
- [ ] Provide DTW as optional "drift-free" alignment mode
- [ ] Compare DTW vs keyframe interpolation accuracy
- [ ] Handle edge cases (extra footage, gaps)

### Phase 10: Performance Optimizations

- [ ] Add multiprocessing.Pool for frame hash computations
- [ ] Parallelize similarity matrix calculations
- [ ] Implement batched frame extraction with threading
- [ ] Add detailed progress reporting with time estimates
- [ ] Profile and optimize memory usage
- [ ] Consider numba JIT for critical loops

### Phase 11: Testing & Validation

- [ ] Create performance benchmark suite
- [ ] Test with tests/bg.mp4 and tests/fg.mp4
- [ ] Measure speedup: target 10-20x improvement
- [ ] Validate zero drift with DTW mode
- [ ] Test various video scenarios (different fps, durations)
- [ ] Add unit tests for new hash/DTW components

### Phase 12: Integration & Documentation

- [ ] Update CLI with new options (--temporal_align dtw)
- [ ] Update CHANGELOG.md with performance improvements
- [ ] Document new alignment modes in README.md
- [ ] Add performance tuning guide
- [ ] Clean up and optimize imports
- [ ] Run final formatting/linting

## Performance Targets

- Current: ~2-5 fps for frame matching (8s video times out)
- Target: 40-80 fps (process 8s video in <1s)
- Zero temporal drift with DTW mode
- Memory usage < 1GB for 1080p videos

## Implementation Priority

1. **Perceptual hashing** (Phase 7) - Biggest impact, easiest
2. **Adaptive windows** (Phase 8) - Quick win, moderate effort
3. **Parallelization** (Phase 10) - Good speedup, moderate effort
4. **DTW** (Phase 9) - Eliminates drift, complex but valuable
