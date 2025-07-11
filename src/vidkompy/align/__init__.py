#!/usr/bin/env python3
# this_file: src/vidkompy/align/__init__.py

"""
Intelligent Thumbnail Detection Package.

This package provides advanced thumbnail detection capabilities with
multi-precision analysis, feature matching, and template correlation.

Main Components:
- ThumbnailFinder: Main orchestrator class
- PrecisionAnalyzer: Multi-level precision analysis
- FrameExtractor: Video/image frame extraction
- ResultDisplayer: Rich console output formatting

Usage:
    from vidkompy.align import ThumbnailFinder

    finder = ThumbnailFinder()
    result = finder.find_thumbnail("fg.mp4", "bg.mp4")
"""

# Core classes
from vidkompy.align.core import ThumbnailFinder
from vidkompy.align.precision import PrecisionAnalyzer
from vidkompy.align.frame_extractor import FrameExtractor
from vidkompy.align.display import ResultDisplayer

# Algorithm classes
from vidkompy.align.algorithms import (
    TemplateMatchingAlgorithm,
    FeatureMatchingAlgorithm,
    SubPixelRefinementAlgorithm,
    PhaseCorrelationAlgorithm,
    HybridMatchingAlgorithm,
)

# Data types
from vidkompy.align.data_types import (
    ThumbnailResult,
    MatchResult,
    PrecisionAnalysisResult,
    AnalysisData,
    FrameExtractionResult,
    PrecisionLevel,
)

# CLI function
from .cli import find_thumbnail

# Version info
__version__ = "1.0.0"
__author__ = "vidkompy"

# Public API
__all__ = [
    "AnalysisData",
    "FeatureMatchingAlgorithm",
    "FrameExtractionResult",
    "FrameExtractor",
    "HybridMatchingAlgorithm",
    "MatchResult",
    "PhaseCorrelationAlgorithm",
    "PrecisionAnalysisResult",
    "PrecisionAnalyzer",
    "PrecisionLevel",
    "ResultDisplayer",
    "SubPixelRefinementAlgorithm",
    # Algorithm classes
    "TemplateMatchingAlgorithm",
    # Core classes
    "ThumbnailFinder",
    # Data types
    "ThumbnailResult",
    # CLI function
    "find_thumbnail",
]


def get_version() -> str:
    """Get package version.

"""
    return __version__


def get_info() -> dict:
    """Get package information.

"""
    return {
        "name": "vidkompy.align",
        "version": __version__,
        "author": __author__,
        "description": "Intelligent thumbnail detection with multi-precision analysis",
    }
