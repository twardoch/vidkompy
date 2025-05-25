# Progress: vidkompy Performance Improvements

## Completed Tasks

### Temporal Alignment Research & Design
- [x] Analyzed current temporal alignment implementation and identified drift issues
- [x] Researched best practices for precise video synchronization
- [x] Created SPEC.md with detailed design for 'precise' engine
- [x] Added --engine CLI parameter to support multiple alignment engines
- [x] Implemented precise temporal alignment engine with multi-resolution approach
- [x] Tested both engines and confirmed performance characteristics
- [x] Updated README.md with comprehensive documentation of both engines
- [x] Added detailed performance benchmarks and usage examples

### Drift Issue Resolution
- [x] Identified root cause of 5-second drift issue in precise engine
- [x] Implemented spatial cropping for temporal alignment (crop background to foreground region)
- [x] Added crop parameter support to VideoProcessor.extract_all_frames()
- [x] Integrated spatial alignment step before temporal alignment in precise engine
- [x] Tuned drift correction parameters (interval: 32→100, blend factor: 0.7→0.85)
- [x] Added drift tracking and logging for debugging
- [x] Verified fix resolves hand movement synchronization issue

### Precise Engine Improvement (Wave Drift Fix - Idea 1 from SPEC.md)
- [x] Updated `PreciseEngineConfig` with new parameters for enhanced drift correction (polynomial model, adaptive blend) and Savitzky-Golay smoothing.
- [x] Modified `MultiResolutionAligner.apply_drift_correction` to use polynomial regression for baseline drift and an adaptive blend factor.
- [x] Added global Savitzky-Golay smoothing pass in `MultiResolutionAligner.align` after drift correction and before full interpolation.
- [x] Adjusted loop variables and conditions in `apply_drift_correction` and `refine_alignment` for clarity and correctness.
- [x] Added safety checks for empty/short mappings in `interpolate_full_mapping` and `align`.

## Future Optimizations (Not Yet Implemented)

### Performance Enhancements
- [ ] GPU acceleration with CuPy for phase correlation
- [ ] FAISS integration for fast similarity search
- [ ] Replace OpenCV with PyAV for faster video I/O
- [ ] Implement sliding window refinement for drift correction (This was part of the precise engine, but could be reviewed/enhanced based on Idea 1's impact)
- [ ] Hierarchical multi-resolution matching (This is part of the precise engine, could be tuned)

### Architecture Improvements
- [ ] Replace template matching with phase correlation for spatial alignment
- [ ] Use neural embeddings (MobileNet) instead of perceptual hashes
- [ ] Implement proper fallback strategies for edge cases
- [ ] Add caching for repeated video pairs
- [ ] Implement Improvement Idea 2 (Optical Flow-Assisted Consistency) from SPEC.md
- [ ] Implement Improvement Idea 3 (Dominant Path DTW) from SPEC.md
- [ ] Fix CLI `-w` (window) parameter bug for `precise` engine (not used effectively).

### Code Quality
- [ ] Add comprehensive unit tests for new drift correction and smoothing.
- [ ] Create performance benchmark suite (or adapt existing `benchmark.sh` to test new config parameters).
- [ ] Add type hints throughout (ongoing).
- [ ] Improve error handling and recovery.
- [ ] Resolve remaining linter errors for line length in `multi_resolution_aligner.py`.