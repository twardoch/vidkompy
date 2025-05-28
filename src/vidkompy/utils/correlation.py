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

__all__ = ["compute_normalized_correlation", "histogram_correlation"]


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
    # Handle NaN inputs
    if np.any(np.isnan(template)) or np.any(np.isnan(image)):
        return 0.0

    template_mean = np.mean(template)
    image_mean = np.mean(image)

    numerator = np.sum((template - template_mean) * (image - image_mean))
    template_var = np.sum((template - template_mean) ** 2)
    image_var = np.sum((image - image_mean) ** 2)

    # Handle zero variance edge cases
    if template_var == 0 or image_var == 0:
        return 0.0

    denominator = np.sqrt(template_var * image_var)
    if denominator == 0:
        return 0.0

    return numerator / denominator


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
    # Handle empty or NaN histograms
    if len(hist1) == 0 or len(hist2) == 0:
        return 0.0
    if np.any(np.isnan(hist1)) or np.any(np.isnan(hist2)):
        return 0.0

    # Normalize histograms with small epsilon to avoid division by zero
    sum1 = np.sum(hist1)
    sum2 = np.sum(hist2)

    if sum1 == 0 or sum2 == 0:
        return 0.0

    h1 = hist1 / (sum1 + 1e-7)
    h2 = hist2 / (sum2 + 1e-7)

    # Compute correlation coefficient
    mean1 = np.mean(h1)
    mean2 = np.mean(h2)

    numerator = np.sum((h1 - mean1) * (h2 - mean2))
    var1 = np.sum((h1 - mean1) ** 2)
    var2 = np.sum((h2 - mean2) ** 2)

    denominator = np.sqrt(var1 * var2)

    if denominator == 0:
        return 0.0

    return numerator / denominator
