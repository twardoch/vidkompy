# Changelog

All notable changes to vidkompy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Spatial alignment results now logged in non-verbose mode for better visibility
- Temporal alignment results now logged in non-verbose mode showing method, offset, and frame count
- Comprehensive docstrings explaining the "why" behind design decisions in all core modules
- SPEC4.md: Detailed performance improvement plan with DTW algorithm and perceptual hashing

### Changed
- Progress indicators improved to prevent flickering between spinner and progress bar
- Progress bars now show time remaining for better user experience
- Progress messages indented for visual hierarchy

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