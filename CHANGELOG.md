# Changelog

All notable changes to vidkompy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Critical Performance**: Fixed compositing bottleneck by implementing sequential reading with generators
- **UI Bug**: Resolved "Only one live display may be active at once" error from nested Progress contexts
- **Compositing Speed**: Eliminated costly random seeks in video files during frame composition
- **Progress UX**: Removed annoying spinner displays for quick operations

### Added
- Sequential frame generators for optimal video reading performance
- Detailed frame composition progress bar with percentage, frame count, and time remaining
- Spatial alignment results now logged in non-verbose mode for better visibility
- Temporal alignment results now logged in non-verbose mode showing method, offset, and frame count
- Comprehensive docstrings explaining the "why" behind design decisions in all core modules
- SPEC4.md: Detailed performance improvement plan with DTW algorithm and perceptual hashing
- Border-based temporal alignment mode with mask generation
- Smooth alpha blending for frame edges
- Sliding window constraint for frame matching optimization

### Changed
- **Performance**: Compositing now uses forward-only sequential reads instead of random seeks (10-100x speedup)
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

### Technical Details
- Removed SpinnerColumn from inner progress bars to prevent conflicts
- Added TimeRemainingColumn for better progress estimation
- Made outer progress transient to reduce visual clutter