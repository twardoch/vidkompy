#!/usr/bin/env python3
# this_file: src/vidkompy/utils/correlation.py

"""
Correlation computation utilities with Numba optimization.

This module provides fast correlation functions used across the
alignment and matching algorithms. All functions are JIT-compiled
with Numba for optimal performance.

Used in:
- vidkompy/align/algorithms.py
- vidkompy/align/precision.py
- vidkompy/comp/numba_optimizations.py
"""

import numpy as np
from numba import jit

# Correlation epsilon to avoid numerical issues
CORR_EPS = 1e-7

__all__ = ["CORR_EPS", "compute_normalized_correlation", "histogram_correlation"]


@jit(nopython=True)
def _safe_corr(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """
    Safe correlation computation with NaN and variance guards.

    Private helper that centralizes the common correlation logic
    used by both public correlation functions.

    Args:
        arr1: First array
        arr2: Second array

    Returns:
        Normalized correlation coefficient between -1 and 1
        Returns 0.0 for edge cases (NaN inputs, zero variance)

    """
    # Handle NaN inputs
    if np.any(np.isnan(arr1)) or np.any(np.isnan(arr2)):
        return 0.0

    mean1 = np.mean(arr1)
    mean2 = np.mean(arr2)

    numerator = np.sum((arr1 - mean1) * (arr2 - mean2))
    var1 = np.sum((arr1 - mean1) ** 2)
    var2 = np.sum((arr2 - mean2) ** 2)

    # Handle zero variance edge cases
    if var1 == 0 or var2 == 0:
        return 0.0

    denominator = np.sqrt(var1 * var2)
    if denominator == 0:
        return 0.0

    return numerator / denominator


@jit(nopython=True)
def compute_normalized_correlation(template: np.ndarray, image: np.ndarray) -> float:
    """
    Fast normalized cross-correlation computation using Numba.

    Computes the normalized cross-correlation between two arrays,
    handling edge cases like zero variance gracefully.

    Args:
        template: Template array to match
        image: Image array to search in

    Returns:
        Normalized correlation coefficient between -1 and 1
        Returns 0.0 for edge cases (zero variance, NaN inputs)

    Used in:
    - vidkompy/align/algorithms.py
    - vidkompy/utils/__init__.py
    """
    return _safe_corr(template, image)


@jit(nopython=True)
def histogram_correlation(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Fast histogram correlation for ballpark scale estimation.

    Computes normalized correlation between two histograms,
    useful for quick scale estimation based on intensity distributions.

    Args:
        hist1: First histogram array
        hist2: Second histogram array

    Returns:
        Correlation coefficient between -1 and 1
        Returns 0.0 for edge cases (empty histograms, zero variance)

    Used in:
    - vidkompy/align/algorithms.py
    - vidkompy/align/precision.py
    - vidkompy/utils/__init__.py
    """
    # Handle empty histograms
    if len(hist1) == 0 or len(hist2) == 0:
        return 0.0

    # Normalize histograms with epsilon to avoid division by zero
    sum1 = np.sum(hist1)
    sum2 = np.sum(hist2)

    if sum1 == 0 or sum2 == 0:
        return 0.0

    h1 = hist1 / (sum1 + CORR_EPS)
    h2 = hist2 / (sum2 + CORR_EPS)

    return _safe_corr(h1, h2)
