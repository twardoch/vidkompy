# Changelog

All notable changes to vidkompy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Critical Performance**: Fixed compositing bottleneck by implementing sequential reading with generators
- **UI Bug**: Resolved "Only one live display may be active at once" error from nested Progress contexts
- **Compositing Speed**: Eliminated costly random seeks in video files during frame composition

### Added
- Sequential frame generators for optimal video reading performance
- Progressive logging for compositing stage with percentage progress updates
- Spatial alignment results now logged in non-verbose mode for better visibility
- Temporal alignment results now logged in non-verbose mode showing method, offset, and frame count
- Comprehensive docstrings explaining the "why" behind design decisions in all core modules
- SPEC4.md: Detailed performance improvement plan with DTW algorithm and perceptual hashing

### Changed
- **Performance**: Compositing now uses forward-only sequential reads instead of random seeks (10-100x speedup)
- Progress indicators improved to prevent flickering between spinner and progress bar  
- Progress bars now show time remaining for better user experience
- Progress messages indented for visual hierarchy
- Compositing progress now logs every 10% instead of using nested Progress bars

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