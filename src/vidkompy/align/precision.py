#!/usr/bin/env python3
# this_file: src/vidkompy/align/precision.py

"""
Multi-precision analysis system for thumbnail detection.

This module implements the progressive refinement system that provides
different levels of speed vs accuracy trade-offs.

"""

import time

import numpy as np

from .result_types import MatchResult, PrecisionLevel, PrecisionAnalysisResult
from .algorithms import (
    TemplateMatchingAlgorithm,
    FeatureMatchingAlgorithm,
    HistogramCorrelationAlgorithm,
    SubPixelRefinementAlgorithm,
    PhaseCorrelationAlgorithm,
    HybridMatchingAlgorithm,
)


class PrecisionAnalyzer:
    """
    Multi-precision analysis system for thumbnail detection.

    This class implements a progressive refinement approach where
    each precision level builds on the previous results to provide
    increasingly accurate matches with corresponding time costs.

    Used in:
    - vidkompy/align/__init__.py
    - vidkompy/align/core.py
    """

    def __init__(self, verbose: bool = False):
        """Initialize the precision analyzer with algorithm instances."""
        self.verbose = verbose
        self.template_matcher = TemplateMatchingAlgorithm(verbose)
        self.feature_matcher = FeatureMatchingAlgorithm(verbose=verbose)
        self.histogram_correlator = HistogramCorrelationAlgorithm()
        self.subpixel_refiner = SubPixelRefinementAlgorithm()
        self.phase_matcher = PhaseCorrelationAlgorithm(verbose)
        self.hybrid_matcher = HybridMatchingAlgorithm(verbose)

    def analyze_at_precision(
        self,
        fg_frame: np.ndarray,
        bg_frame: np.ndarray,
        precision_level: int,
        previous_results: list[PrecisionAnalysisResult] | None = None,
    ) -> PrecisionAnalysisResult:
        """
        Perform analysis at a specific precision level.

        Args:
            fg_frame: Foreground frame
            bg_frame: Background frame
            precision_level: Precision level (0-4)
            previous_results: Results from previous precision levels

        Returns:
            PrecisionAnalysisResult for this level

        """
        level = PrecisionLevel(precision_level)
        start_time = time.time()

        if level == PrecisionLevel.BALLPARK:
            result = self._analyze_ballpark(fg_frame, bg_frame)
        elif level == PrecisionLevel.COARSE:
            result = self._analyze_coarse(fg_frame, bg_frame, previous_results)
        elif level == PrecisionLevel.BALANCED:
            result = self._analyze_balanced(fg_frame, bg_frame, previous_results)
        elif level == PrecisionLevel.FINE:
            result = self._analyze_fine(fg_frame, bg_frame, previous_results)
        elif level == PrecisionLevel.PRECISE:
            result = self._analyze_precise(fg_frame, bg_frame, previous_results)
        else:
            msg = f"Invalid precision level: {precision_level}"
            raise ValueError(msg)

        processing_time = time.time() - start_time

        return PrecisionAnalysisResult(
            level=level,
            scale=result.scale,
            x=result.x,
            y=result.y,
            confidence=result.confidence,
            processing_time=processing_time,
            method=result.method,
        )

    def progressive_analysis(
        self, fg_frame: np.ndarray, bg_frame: np.ndarray, max_precision: int = 2
    ) -> list[PrecisionAnalysisResult]:
        """
        Perform progressive analysis up to specified precision level.

        Args:
            fg_frame: Foreground frame
            bg_frame: Background frame
            max_precision: Maximum precision level to reach

        Returns:
            List of PrecisionAnalysisResult for each level

        Used in:
        - vidkompy/align/core.py
        """
        results = []

        for level in range(max_precision + 1):
            result = self.analyze_at_precision(fg_frame, bg_frame, level, results)
            results.append(result)

        return results

    def _analyze_ballpark(
        self, fg_frame: np.ndarray, bg_frame: np.ndarray
    ) -> MatchResult:
        """
        Level 0: Ultra-fast ballpark estimate using advanced histogram correlation.

        This provides a rough scale estimate in ~1ms by comparing
        color histograms with the enhanced algorithm from template matcher.

        """
        scale, confidence = self.template_matcher.ballpark_scale_estimation(
            fg_frame, bg_frame, scale_range=(0.3, 1.5)
        )

        return MatchResult(
            confidence=confidence,
            x=0,  # No position estimate at this level
            y=0,
            scale=scale,
            method="ballpark_enhanced",
        )

    def _analyze_coarse(
        self,
        fg_frame: np.ndarray,
        bg_frame: np.ndarray,
        previous_results: list[PrecisionAnalysisResult],
    ) -> MatchResult:
        """
        Level 1: Coarse template matching with parallel processing.

        Uses the ballpark scale estimate to focus template matching
        in a narrower range with parallel processing.

        """
        # Get ballpark scale estimate
        ballpark_scale = 1.0
        if previous_results:
            ballpark_scale = previous_results[0].scale

        # Define search range around ballpark estimate
        scale_range = (max(0.3, ballpark_scale * 0.8), min(1.5, ballpark_scale * 1.2))

        # Use parallel template matching for better performance
        result = self.template_matcher.parallel_multiscale_template_matching(
            fg_frame, bg_frame, scale_range, scale_steps=10
        )

        if result:
            return result
        else:
            return MatchResult(0.0, 0, 0, ballpark_scale, method="coarse")

    def _analyze_balanced(
        self,
        fg_frame: np.ndarray,
        bg_frame: np.ndarray,
        previous_results: list[PrecisionAnalysisResult],
    ) -> MatchResult:
        """
        Level 2: Balanced feature + template matching.

        Combines feature matching with refined template matching
        for good accuracy in reasonable time.

        """
        # Try enhanced feature matching first
        feature_result = self.feature_matcher.enhanced_feature_matching(
            fg_frame, bg_frame, detector_type="auto"
        )

        # Get coarse template result for comparison and use ballpark estimate if available
        ballpark_scale = 1.0
        if previous_results:
            if len(previous_results) >= 1:
                ballpark_scale = previous_results[0].scale
            if len(previous_results) >= 2:
                previous_results[1].scale

        # Use ballpark estimation for template matching
        template_result = self.template_matcher.parallel_multiscale_template_matching(
            fg_frame,
            bg_frame,
            scale_range=(
                max(0.3, ballpark_scale * 0.8),
                min(1.5, ballpark_scale * 1.2),
            ),
            scale_steps=15,
        )

        # Choose best result
        candidates = [r for r in [feature_result, template_result] if r is not None]

        if candidates:
            return max(candidates, key=lambda r: r.confidence)
        # Fallback to previous result
        elif previous_results:
            prev = previous_results[-1]
            return MatchResult(
                prev.confidence, prev.x, prev.y, prev.scale, method="balanced"
            )
        else:
            return MatchResult(0.0, 0, 0, 1.0, method="balanced")

    def _analyze_fine(
        self,
        fg_frame: np.ndarray,
        bg_frame: np.ndarray,
        previous_results: list[PrecisionAnalysisResult],
    ) -> MatchResult:
        """
        Level 3: Fine hybrid matching with multiple algorithms.

        Uses the hybrid matching algorithm that combines feature matching,
        template matching, and phase correlation for high quality results.

        """
        # Use hybrid matching for best results
        hybrid_result = self.hybrid_matcher.hybrid_matching(
            fg_frame, bg_frame, method="accurate"
        )

        if hybrid_result:
            return hybrid_result

        # Fallback to enhanced feature matching
        feature_result = self.feature_matcher.enhanced_feature_matching(
            fg_frame, bg_frame, detector_type="auto"
        )

        if feature_result:
            return feature_result

        # Final fallback
        if previous_results:
            prev = previous_results[-1]
            return MatchResult(
                prev.confidence, prev.x, prev.y, prev.scale, method="fine"
            )
        else:
            return MatchResult(0.0, 0, 0, 1.0, method="fine")

    def _analyze_precise(
        self,
        fg_frame: np.ndarray,
        bg_frame: np.ndarray,
        previous_results: list[PrecisionAnalysisResult],
    ) -> MatchResult:
        """
        Level 4: Precise sub-pixel refinement.

        Takes the best previous result and refines it with
        sub-pixel accuracy for maximum precision.

        """
        # Get fine result as starting point
        fine_result = None
        if len(previous_results) >= 4:
            fine = previous_results[3]
            fine_result = MatchResult(
                fine.confidence, fine.x, fine.y, fine.scale, fine.method
            )
        elif previous_results:
            # Use best available result
            best_prev = max(previous_results, key=lambda r: r.confidence)
            fine_result = MatchResult(
                best_prev.confidence,
                best_prev.x,
                best_prev.y,
                best_prev.scale,
                best_prev.method,
            )

        if fine_result is None:
            return MatchResult(0.0, 0, 0, 1.0, method="precise")

        # Apply sub-pixel refinement
        refined_result = self.subpixel_refiner.refine_position(
            fg_frame, bg_frame, fine_result
        )

        return refined_result

    def get_scale_range_for_level(
        self, level: PrecisionLevel, previous_scale: float = 1.0
    ) -> tuple[float, float]:
        """
        Get appropriate scale range for a precision level.

        Args:
            level: Precision level
            previous_scale: Scale estimate from previous level

        Returns:
            Tuple of (min_scale, max_scale)

        """
        if level == PrecisionLevel.BALLPARK:
            return (0.3, 1.5)
        elif level == PrecisionLevel.COARSE:
            return (max(0.3, previous_scale * 0.8), min(1.5, previous_scale * 1.2))
        elif level == PrecisionLevel.BALANCED:
            return (max(0.3, previous_scale * 0.9), min(1.5, previous_scale * 1.1))
        elif level == PrecisionLevel.FINE:
            return (max(0.3, previous_scale * 0.95), min(1.5, previous_scale * 1.05))
        elif level == PrecisionLevel.PRECISE:
            return (max(0.3, previous_scale * 0.98), min(1.5, previous_scale * 1.02))
        else:
            return (0.5, 1.5)

    def get_scale_steps_for_level(self, level: PrecisionLevel) -> int:
        """
        Get appropriate number of scale steps for a precision level.

        Args:
            level: Precision level

        Returns:
            Number of scale steps to use

        """
        steps_map = {
            PrecisionLevel.BALLPARK: 10,
            PrecisionLevel.COARSE: 5,
            PrecisionLevel.BALANCED: 10,
            PrecisionLevel.FINE: 20,
            PrecisionLevel.PRECISE: 30,
        }
        return steps_map.get(level, 10)
