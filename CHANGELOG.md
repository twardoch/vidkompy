# Changelog

All notable changes to vidkompy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Implemented _precompute_masked_video_fingerprints() for border mode DTW support
- Added _compute_masked_frame_hash() for efficient masked perceptual hashing
- Removed SpinnerColumn from inner progress bars to prevent conflicts
- Added TimeRemainingColumn for better progress estimation
- Made outer progress transient to reduce visual clutter