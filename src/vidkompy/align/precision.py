#!/usr/bin/env python3
# this_file: src/vidkompy/align/precision.py

"""
Multi-precision analysis system for thumbnail detection.

This module implements the progressive refinement system that provides
different levels of speed vs accuracy trade-offs.

"""

import time
from collections import namedtuple
from collections.abc import Callable
from functools import cached_property

import numpy as np

from vidkompy.align.data_types import (
    MatchResult,
    PrecisionLevel,
    PrecisionAnalysisResult,
)
from vidkompy.align.algorithms import (
    TemplateMatchingAlgorithm,
    FeatureMatchingAlgorithm,
    SubPixelRefinementAlgorithm,
    PhaseCorrelationAlgorithm,
    HybridMatchingAlgorithm,
)


# Scale parameters for each precision level using namedtuple for type safety
ScaleParams = namedtuple("ScaleParams", ["range_fn", "steps"])


def _ballpark_range(prev_scale: float = 1.0) -> tuple[float, float]:
    """Wide search range for ballpark estimation.

"""
    return (0.3, 1.5)


def _coarse_range(prev_scale: float = 1.0) -> tuple[float, float]:
    """Narrow around previous estimate.

"""
    return (max(0.3, prev_scale * 0.8), min(1.5, prev_scale * 1.2))


def _balanced_range(prev_scale: float = 1.0) -> tuple[float, float]:
    """Balanced refinement around previous estimate.

"""
    return (max(0.3, prev_scale * 0.9), min(1.5, prev_scale * 1.1))


def _fine_range(prev_scale: float = 1.0) -> tuple[float, float]:
    """Fine-grained search around previous estimate.

"""
    return (max(0.3, prev_scale * 0.95), min(1.5, prev_scale * 1.05))


def _precise_range(prev_scale: float = 1.0) -> tuple[float, float]:
    """Very precise search around previous estimate.

"""
    return (max(0.3, prev_scale * 0.98), min(1.5, prev_scale * 1.02))


SCALE_PARAMS: dict[PrecisionLevel, ScaleParams] = {
    PrecisionLevel.BALLPARK: ScaleParams(_ballpark_range, 10),
    PrecisionLevel.COARSE: ScaleParams(_coarse_range, 5),
    PrecisionLevel.BALANCED: ScaleParams(_balanced_range, 10),
    PrecisionLevel.FINE: ScaleParams(_fine_range, 20),
    PrecisionLevel.PRECISE: ScaleParams(_precise_range, 30),
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
        """Initialize the precision analyzer with lazy algorithm loading.

"""
        self.verbose = verbose
        self._algorithms: dict[str, any] = {}  # Lazy loading cache

        # Strategy map for precision analysis
        self._strategy_map: dict[PrecisionLevel, Callable] = {
            PrecisionLevel.BALLPARK: self._analyze_ballpark,
            PrecisionLevel.COARSE: self._analyze_coarse,
            PrecisionLevel.BALANCED: self._analyze_balanced,
            # For MVP, FINE and PRECISE will reuse BALANCED analysis
            PrecisionLevel.FINE: self._analyze_balanced, # self._analyze_fine, # Deferred post-MVP
            PrecisionLevel.PRECISE: self._analyze_balanced, # self._analyze_precise, # Deferred post-MVP
        }

    @cached_property
    def template_matcher(self) -> TemplateMatchingAlgorithm:
        """Template matching algorithm instance.

"""
        return TemplateMatchingAlgorithm(self.verbose)

    @cached_property
    def feature_matcher(self) -> FeatureMatchingAlgorithm:
        """Feature matching algorithm instance.

"""
        return FeatureMatchingAlgorithm(verbose=self.verbose)

    @cached_property
    def subpixel_refiner(self) -> SubPixelRefinementAlgorithm:
        """Sub-pixel refinement algorithm instance.

"""
        return SubPixelRefinementAlgorithm()

    @cached_property
    def phase_matcher(self) -> PhaseCorrelationAlgorithm:
        """Phase correlation algorithm instance.

"""
        return PhaseCorrelationAlgorithm(self.verbose)

    @cached_property
    def hybrid_matcher(self) -> HybridMatchingAlgorithm:
        """Hybrid matching algorithm instance.

"""
        return HybridMatchingAlgorithm(self.verbose)

    def _get_algorithm(self, name: str):
        """Legacy method for backward compatibility.

"""
        if name == "template_matcher":
            return self.template_matcher
        elif name == "feature_matcher":
            return self.feature_matcher
        elif name == "subpixel_refiner":
            return self.subpixel_refiner
        elif name == "phase_matcher":
            return self.phase_matcher
        elif name == "hybrid_matcher":
            return self.hybrid_matcher
        else:
            msg = f"Unknown algorithm: {name}"
            raise NotImplementedError(msg)

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
        # For MVP, we effectively cap detailed analysis at level 2 (BALANCED).
        # Higher requested levels will reuse level 2's results.
        mvp_max_precision_stages = min(max_precision, PrecisionLevel.BALANCED.value)

        for level_val in range(mvp_max_precision_stages + 1):
            result = self.analyze_at_precision(fg_frame, bg_frame, level_val, results)
            results.append(result)

        # If higher precision was requested, fill with results from the best MVP stage
        if max_precision > mvp_max_precision_stages and results:
            best_mvp_result = results[-1] # Last computed result is from BALANCED or lower
            for level_val in range(mvp_max_precision_stages + 1, max_precision + 1):
                # Create a new result object indicating it's a carry-over
                carried_over_result = PrecisionAnalysisResult(
                    level=PrecisionLevel(level_val),
                    scale=best_mvp_result.scale,
                    x=best_mvp_result.x,
                    y=best_mvp_result.y,
                    confidence=best_mvp_result.confidence,
                    processing_time=0.0, # No additional processing time
                    method=f"{best_mvp_result.method}_mvp_carry_over"
                )
                results.append(carried_over_result)

        return results

    def _analyze_ballpark(
        self, fg_frame: np.ndarray, bg_frame: np.ndarray
    ) -> MatchResult:
        """
        Level 0: Ultra-fast ballpark estimate using advanced histogram correlation.

        This provides a rough scale estimate in ~1ms by comparing
        color histograms with the enhanced algorithm from template matcher.

        """
        scale_params = SCALE_PARAMS[PrecisionLevel.BALLPARK]
        scale_range = scale_params.range_fn()

        scale, confidence = self.template_matcher.ballpark_scale_estimation(
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
        scale_params = SCALE_PARAMS[PrecisionLevel.COARSE]
        scale_range = scale_params.range_fn(ballpark_scale)

        # Use parallel template matching for better performance
        result = self.template_matcher.parallel_multiscale_template_matching(
            fg_frame, bg_frame, scale_range, scale_steps=scale_params.steps
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
        # For MVP, enhanced_feature_matching defaults to ORB.
        feature_result = self.feature_matcher.enhanced_feature_matching(
            fg_frame, bg_frame
        )

        # Get coarse template result for comparison and use ballpark estimate if available
        ballpark_scale = 1.0
        if previous_results:
            if len(previous_results) >= 1:
                ballpark_scale = previous_results[0].scale

        # Use ballpark estimation for template matching
        scale_params = SCALE_PARAMS[PrecisionLevel.BALANCED]
        scale_range = scale_params.range_fn(ballpark_scale)

        template_result = self.template_matcher.parallel_multiscale_template_matching(
            fg_frame,
            bg_frame,
            scale_range=scale_range,
            scale_steps=scale_params.steps,
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
        # For MVP, enhanced_feature_matching defaults to ORB.
        feature_result = self.feature_matcher.enhanced_feature_matching(
            fg_frame, bg_frame
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
