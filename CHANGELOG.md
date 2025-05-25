# Changelog

All notable changes to vidkompy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed - Major Engine Simplification

- **Fast Engine**: Completely removed the 'fast' engine and all DTW-based alignment functionality
  - Removed \_align_frames_dtw method and related keyframe matching code
  - Removed perceptual hashing methods for old engines
  - Removed cost matrix building and optimal path finding algorithms
  - Removed frame alignment building and interpolation methods
- **Precise Engine**: Completely removed the 'precise' engine and multi-resolution alignment
  - Removed precise_temporal_alignment.py module integration
  - Removed multi-resolution pyramid processing
  - Removed keyframe anchoring and bidirectional DTW
  - Removed sliding window refinement functionality
- **Mask Engine**: Removed the 'mask' engine (was part of precise engine system)
  - Removed masked alignment functionality from precise engine
  - Cleaned up border-mode specific code for old engines

### Changed - Engine Renaming & Defaults

- **Engine Names**: Renamed tunnel engines for simplicity
  - `tunnel_full` → `full` (now the default engine)
  - `tunnel_mask` → `mask`
- **Default Parameters**: Updated optimal defaults based on performance testing
  - Changed drift interval default from 100 to 10 (optimal for tunnel engines)
  - Changed window default from 0 to 10 (optimal performance)
- **CLI Validation**: Updated engine validation to only accept 'full' and 'mask'
- **Documentation**: Updated all engine references in CLI help and comments

### Added - Previous Features

- **Tunnel-Based Temporal Alignment Engines**: Implemented two new alignment engines based on direct frame comparison
  - `full` (formerly tunnel_full): Uses full frame pixel comparison with sliding window approach
  - `mask` (formerly tunnel_mask): Uses masked frame comparison focusing on content regions
  - Both engines perform bidirectional matching (forward and backward passes)
  - Monotonicity enforced by design through sliding window constraints
  - Configurable window size, downsampling, and early stopping thresholds
  - Designed to eliminate temporal drift through direct frame matching

### Added

- **Numba JIT Optimization**: Integrated Numba JIT compilation for performance-critical operations
  - Added `numba>=0.58.0` dependency to `pyproject.toml`
  - Created `numba_optimizations.py` module with optimized implementations
  - DTW cost matrix computation now 5-20x faster with parallelized distance calculations
  - Frame fingerprint batch comparisons now 3-10x faster with vectorized operations
  - Multi-resolution drift correction optimized with JIT-compiled polynomial fitting
  - Automatic fallback to standard implementation if Numba compilation fails
  - First-run compilation overhead mitigated by caching compiled functions

### Changed

- **DTW Algorithm Performance**: Optimized DTWAligner with Numba JIT compilation

  - `_build_dtw_matrix()` uses parallel computation for large alignments
  - `_find_optimal_path()` uses optimized backtracking algorithm
  - Added feature extraction method for converting fingerprints to numpy arrays
  - Intelligent switching between Numba and standard implementation based on problem size

- **Frame Fingerprinting Performance**: Enhanced FrameFingerprinter with batch operations

  - Added `compare_fingerprints_batch()` method for parallel fingerprint comparisons
  - Optimized histogram correlation computation with Numba
  - Weighted similarity calculation now JIT-compiled
  - Batch Hamming distance computation parallelized across multiple cores

- **Multi-Resolution Alignment**: Accelerated drift correction in precise engine

  - Polynomial drift correction now uses Numba-optimized implementation
  - Adaptive blend factor calculation accelerated
  - Monotonicity enforcement optimized with compiled loops

- **Precise Engine Enhancement**: Implemented Idea 1 from `SPEC.md` for the `precise` temporal alignment engine.
  - Updated `PreciseEngineConfig` in `multi_resolution_aligner.py` to include new parameters for enhanced drift correction (polynomial model, adaptive blend factor) and Savitzky-Golay smoothing.
  - Modified `MultiResolutionAligner.apply_drift_correction` to use polynomial regression as a baseline for drift correction and to incorporate an adaptive blend factor, offering more nuanced adjustments than simple linear interpolation.
  - Added a global Savitzky-Golay smoothing pass in `MultiResolutionAligner.align` after drift correction and before the final interpolation to full resolution. This aims to reduce high-frequency oscillations ("flag wave" effect) in the temporal mapping.
  - Improved internal logic in `apply_drift_correction` and `refine_alignment` for clarity and robustness, including better handling of segment boundaries and loop variables.
  - Added safety checks in `interpolate_full_mapping` and `align` to handle empty or very short mappings, preventing potential errors.

### Fixed

- **Parameter Mismatch**: Fixed DTWAligner constructor parameter name from 'window' to 'window' in precise alignment engines to resolve TypeError during precise alignment initialization
- **Temporal Drift Issue**: Fixed severe drift at 5-second mark by implementing spatial cropping during temporal alignment and adjusting drift correction parameters
- **Spatial Cropping**: Added background frame cropping to match foreground region during temporal alignment, eliminating false dissimilarities from non-overlapping areas
- **Drift Correction Tuning**: Increased drift correction interval from 32 to 100 frames and blend factor from 0.7 to 0.85 to prevent overcorrection

### Added

- **Multi-Engine Architecture**: Added --engine CLI parameter to support multiple temporal alignment engines
- **Precise Engine Implementation**: Implemented advanced multi-resolution temporal alignment engine with drift correction
- **Engine Validation**: Added proper validation and error messages for engine selection
- **Comprehensive Engine Documentation**: Updated README.md with detailed explanations of both fast and precise engines
- **Performance Benchmarks**: Added real-world performance comparison data showing 2x vs 40x real-time processing
- **Usage Examples**: Added multiple CLI examples demonstrating different engine configurations

### Major Refactoring: Code Simplification & Performance

### Removed

- **Audio Alignment**: Removed all audio-based temporal alignment functionality (align_audio, extract_audio)
- **Feature Matching**: Removed ORB feature-based spatial alignment method
- **FAST Mode**: Removed audio-first temporal alignment mode
- **CLI Complexity**: Simplified CLI to essential parameters only
- **Redundant Options**: Removed match_time, match_space, temporal_align, skip_spatial_align, trim, max_keyframes, and window parameters
- **Dependencies**: Removed soundfile and scipy.signal as no longer needed

### Changed

- **Fixed Configuration**: Always uses border mode with DTW and template matching
- **Simplified API**: Reduced CLI to just bg, fg, output, border, blend, gpu, and verbose
- **Single Path**: Each operation now has only one implementation path
- **Code Reduction**: Approximately 40% reduction in codebase size

### Implementation Status vs Original Specs

- **SPEC5**: Fully implemented all quick wins and drift elimination features
- **SPEC4**: Partially implemented - removed alternative methods but kept optimized versions instead of full rewrite with FAISS/phase correlation

### Performance Improvements from SPEC5 Implementation

### Fixed

- **Temporal Drift**: Implemented adaptive keyframe density to prevent drift based on FPS differences
- **Border Mode Performance**: Enabled DTW with masked perceptual hashing for fast border mode alignment
- **Critical Performance**: Fixed compositing bottleneck by implementing sequential reading with generators
- **UI Bug**: Resolved "Only one live display may be active at once" error from nested Progress contexts
- **Compositing Speed**: Eliminated costly random seeks in video files during frame composition
- **Progress UX**: Removed annoying spinner displays for quick operations

### Added

- **Masked Fingerprinting**: Added compute_masked_fingerprint() method for border-aware perceptual hashing
- **Adaptive Keyframes**: calculate_adaptive_keyframe_count() adjusts density based on video characteristics
- **Benchmark Suite**: Created benchmark.py for comprehensive performance testing
- **Border Mode + DTW**: DTW temporal alignment now works with masked regions
- Sequential frame generators for optimal video reading performance
- Detailed frame composition progress bar with percentage, frame count, and time remaining
- Spatial alignment results now logged in non-verbose mode for better visibility
- Temporal alignment results now logged in non-verbose mode showing method, offset, and frame count
- Comprehensive docstrings explaining the "why" behind design decisions in all core modules
- SPEC4.md: Detailed performance improvement plan with DTW algorithm and perceptual hashing
- SPEC5.md: Drift elimination and performance optimization specification
- Border-based temporal alignment mode with mask generation
- Smooth alpha blending for frame edges
- Sliding window constraint for frame matching optimization

### Changed

- **Default Keyframes**: Reduced from 2000 to 200 for better drift prevention
- **Parallel Processing**: Cost matrix building already uses ThreadPoolExecutor for parallel computation
- **Border Mode Logic**: DTW no longer falls back to classic alignment in border mode
- **Performance**: Compositing now uses forward-only sequential reads instead of random seeks (10-100x speedup)
- **Performance**: Significantly reduced keyframe sampling when using SSIM (e.g., in border mode fallback), drastically improving speed for that specific path
- **Progress UX**: Quick tasks (video analysis, spatial alignment) now use simple logging instead of spinners
- **Progress Bars**: Frame composition shows meaningful progress bar instead of percentage logging
- **Default Mode**: Border-based temporal alignment is now the default for improved accuracy
- Progress bars now show time remaining for better user experience
- Maintained useful progress bars for time-intensive operations (DTW, cost matrix building)

### Documentation

- Added detailed docstrings to alignment_engine.py explaining architecture decisions
- Added detailed docstrings to spatial_alignment.py explaining algorithm choices
- Added detailed docstrings to temporal_alignment.py explaining current limitations
- Added detailed docstrings to video_processor.py explaining tool choices
- Created SPEC4.md with comprehensive improvement plan addressing performance and quality issues
- Created SPEC5.md with drift elimination strategies and performance targets

### Technical Details

- Implemented \_precompute_masked_video_fingerprints() for border mode DTW support
- Added \_compute_masked_frame_hash() for efficient masked perceptual hashing
- Removed SpinnerColumn from inner progress bars to prevent conflicts
- Added TimeRemainingColumn for better progress estimation
- Made outer progress transient to reduce visual clutter
