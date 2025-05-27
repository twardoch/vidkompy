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
)


class PrecisionAnalyzer:
    """
    Multi-precision analysis system for thumbnail detection.

    This class implements a progressive refinement approach where
    each precision level builds on the previous results to provide
    increasingly accurate matches with corresponding time costs.
    """

    def __init__(self):
        """Initialize the precision analyzer with algorithm instances."""
        self.template_matcher = TemplateMatchingAlgorithm()
        self.feature_matcher = FeatureMatchingAlgorithm()
        self.histogram_correlator = HistogramCorrelationAlgorithm()
        self.subpixel_refiner = SubPixelRefinementAlgorithm()

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
        Level 0: Ultra-fast ballpark estimate using histogram correlation.

        This provides a rough scale estimate in ~1ms by comparing
        color histograms without spatial information.
        """
        scale, confidence = self.histogram_correlator.estimate_scale(
            fg_frame, bg_frame, scale_range=(0.3, 1.5), scale_steps=10
        )

        return MatchResult(
            confidence=confidence,
            x=0,  # No position estimate at this level
            y=0,
            scale=scale,
            method="histogram",
        )

    def _analyze_coarse(
        self,
        fg_frame: np.ndarray,
        bg_frame: np.ndarray,
        previous_results: list[PrecisionAnalysisResult],
    ) -> MatchResult:
        """
        Level 1: Coarse template matching with wide scale steps.

        Uses the ballpark scale estimate to focus template matching
        in a narrower range with ~20% steps.
        """
        # Get ballpark scale estimate
        ballpark_scale = 1.0
        if previous_results:
            ballpark_scale = previous_results[0].scale

        # Define search range around ballpark estimate
        scale_range = (max(0.3, ballpark_scale * 0.8), min(1.5, ballpark_scale * 1.2))

        # Perform template matching with coarse steps
        results = self.template_matcher.match_at_multiple_scales(
            fg_frame, bg_frame, scale_range, scale_steps=5, method="coarse"
        )

        # Return best result
        if results:
            return max(results, key=lambda r: r.confidence)
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
        # Try feature matching first
        feature_result = self.feature_matcher.match_features(
            fg_frame, bg_frame, method="feature"
        )

        # Get coarse template result for comparison
        template_result = None
        if len(previous_results) >= 2:
            coarse = previous_results[1]
            # Refine around coarse result
            scale_range = (max(0.3, coarse.scale * 0.9), min(1.5, coarse.scale * 1.1))
            template_results = self.template_matcher.match_at_multiple_scales(
                fg_frame, bg_frame, scale_range, scale_steps=10, method="balanced"
            )
            if template_results:
                template_result = max(template_results, key=lambda r: r.confidence)

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
        Level 3: Fine feature + focused template matching.

        Uses enhanced feature matching and very focused template
        search for high quality results.
        """
        # Get balanced result as starting point
        balanced_result = None
        if len(previous_results) >= 3:
            balanced = previous_results[2]
            balanced_result = MatchResult(
                balanced.confidence,
                balanced.x,
                balanced.y,
                balanced.scale,
                balanced.method,
            )

        # Enhanced feature matching with more features
        enhanced_feature_matcher = FeatureMatchingAlgorithm(max_features=1000)
        feature_result = enhanced_feature_matcher.match_features(
            fg_frame, bg_frame, method="fine_feature"
        )

        # Very focused template matching
        template_result = None
        if balanced_result:
            scale_range = (
                max(0.3, balanced_result.scale * 0.95),
                min(1.5, balanced_result.scale * 1.05),
            )
            template_results = self.template_matcher.match_at_multiple_scales(
                fg_frame, bg_frame, scale_range, scale_steps=20, method="fine"
            )
            if template_results:
                template_result = max(template_results, key=lambda r: r.confidence)

        # Choose best result
        candidates = [
            r
            for r in [feature_result, template_result, balanced_result]
            if r is not None
        ]

        if candidates:
            return max(candidates, key=lambda r: r.confidence)
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
