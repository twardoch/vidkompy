#!/usr/bin/env python3
# this_file: src/vidkompy/align/precision.py

"""
Multi-precision analysis system for thumbnail detection.

This module implements the progressive refinement system that provides
different levels of speed vs accuracy trade-offs.

"""

import time
from collections.abc import Callable

import numpy as np

from .result_types import MatchResult, PrecisionLevel, PrecisionAnalysisResult
from .algorithms import (
    TemplateMatchingAlgorithm,
    FeatureMatchingAlgorithm,
    SubPixelRefinementAlgorithm,
    PhaseCorrelationAlgorithm,
    HybridMatchingAlgorithm,
)


# Scale parameters for each precision level: (scale_range_func, scale_steps)
SCALE_PARAMS: dict[PrecisionLevel, tuple[Callable, int]] = {
    PrecisionLevel.BALLPARK: (lambda prev_scale=1.0: (0.3, 1.5), 10),
    PrecisionLevel.COARSE: (
        lambda prev_scale=1.0: (max(0.3, prev_scale * 0.8), min(1.5, prev_scale * 1.2)),
        5,
    ),
    PrecisionLevel.BALANCED: (
        lambda prev_scale=1.0: (max(0.3, prev_scale * 0.9), min(1.5, prev_scale * 1.1)),
        10,
    ),
    PrecisionLevel.FINE: (
        lambda prev_scale=1.0: (
            max(0.3, prev_scale * 0.95),
            min(1.5, prev_scale * 1.05),
        ),
        20,
    ),
    PrecisionLevel.PRECISE: (
        lambda prev_scale=1.0: (
            max(0.3, prev_scale * 0.98),
            min(1.5, prev_scale * 1.02),
        ),
        30,
    ),
}


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
        """Initialize the precision analyzer with lazy algorithm loading."""
        self.verbose = verbose
        self._algorithms: dict[str, any] = {}  # Lazy loading cache

        # Strategy map for precision analysis
        self._strategy_map: dict[PrecisionLevel, Callable] = {
            PrecisionLevel.BALLPARK: self._analyze_ballpark,
            PrecisionLevel.COARSE: self._analyze_coarse,
            PrecisionLevel.BALANCED: self._analyze_balanced,
            PrecisionLevel.FINE: self._analyze_fine,
            PrecisionLevel.PRECISE: self._analyze_precise,
        }

    def _get_algorithm(self, name: str):
        """Lazy singleton access to algorithms."""
        if name not in self._algorithms:
            if name == "template_matcher":
                self._algorithms[name] = TemplateMatchingAlgorithm(self.verbose)
            elif name == "feature_matcher":
                self._algorithms[name] = FeatureMatchingAlgorithm(verbose=self.verbose)
            elif name == "subpixel_refiner":
                self._algorithms[name] = SubPixelRefinementAlgorithm()
            elif name == "phase_matcher":
                self._algorithms[name] = PhaseCorrelationAlgorithm(self.verbose)
            elif name == "hybrid_matcher":
                self._algorithms[name] = HybridMatchingAlgorithm(self.verbose)
            else:
                msg = f"Unknown algorithm: {name}"
                raise ValueError(msg)
        return self._algorithms[name]

    def analyze_at_precision(
        self,
        fg_frame: np.ndarray,
        bg_frame: np.ndarray,
        precision_level: int,
        previous_results: list[PrecisionAnalysisResult] | None = None,
    ) -> PrecisionAnalysisResult:
        """
        Perform analysis at a specific precision level using strategy map.

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

        # Use strategy map to select analysis method
        if level not in self._strategy_map:
            msg = f"Invalid precision level: {precision_level}"
            raise ValueError(msg)

        strategy = self._strategy_map[level]

        # Call appropriate analysis method
        if level == PrecisionLevel.BALLPARK:
            result = strategy(fg_frame, bg_frame)
        else:
            result = strategy(fg_frame, bg_frame, previous_results)

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
        template_matcher = self._get_algorithm("template_matcher")
        scale_range_func, _ = SCALE_PARAMS[PrecisionLevel.BALLPARK]
        scale_range = scale_range_func()

        scale, confidence = template_matcher.ballpark_scale_estimation(
            fg_frame, bg_frame, scale_range=scale_range
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

        # Get scale parameters from SCALE_PARAMS
        scale_range_func, scale_steps = SCALE_PARAMS[PrecisionLevel.COARSE]
        scale_range = scale_range_func(ballpark_scale)

        # Use parallel template matching for better performance
        template_matcher = self._get_algorithm("template_matcher")
        result = template_matcher.parallel_multiscale_template_matching(
            fg_frame, bg_frame, scale_range, scale_steps=scale_steps
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
        feature_matcher = self._get_algorithm("feature_matcher")
        feature_result = feature_matcher.enhanced_feature_matching(
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
        scale_range_func, scale_steps = SCALE_PARAMS[PrecisionLevel.BALANCED]
        scale_range = scale_range_func(ballpark_scale)

        template_matcher = self._get_algorithm("template_matcher")
        template_result = template_matcher.parallel_multiscale_template_matching(
            fg_frame,
            bg_frame,
            scale_range=scale_range,
            scale_steps=scale_steps,
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
        hybrid_matcher = self._get_algorithm("hybrid_matcher")
        hybrid_result = hybrid_matcher.hybrid_matching(
            fg_frame, bg_frame, method="accurate"
        )

        if hybrid_result:
            return hybrid_result

        # Fallback to enhanced feature matching
        feature_matcher = self._get_algorithm("feature_matcher")
        feature_result = feature_matcher.enhanced_feature_matching(
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
        subpixel_refiner = self._get_algorithm("subpixel_refiner")
        refined_result = subpixel_refiner.refine_position(
            fg_frame, bg_frame, fine_result
        )

        return refined_result
