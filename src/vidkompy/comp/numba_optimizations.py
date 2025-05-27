#!/usr/bin/env python3
# this_file: src/vidkompy/comp/numba_optimizations.py

"""
Numba-optimized functions for vidkompy performance bottlenecks.

These JIT-compiled functions provide significant speedups for
computationally intensive operations like DTW and fingerprint comparisons.
"""

import numpy as np
from numba import jit, prange
from loguru import logger


# ==============================================================================
# DTW Algorithm Optimizations (5-20x speedup)
# ==============================================================================


@jit(nopython=True, parallel=True, cache=True)
def compute_dtw_cost_matrix_numba(
    fg_features: np.ndarray, bg_features: np.ndarray, window: int
) -> np.ndarray:
    """
    Optimized DTW cost matrix computation using Numba.

    This replaces the nested loops in DTWAligner._build_dtw_matrix()
    with a parallelized JIT-compiled version.

    Args:
        fg_features: Foreground feature vectors (N, D)
        bg_features: Background feature vectors (M, D)
        window: Sakoe-Chiba band width

    Returns:
        DTW cost matrix (N+1, M+1)

    """
    n_fg, n_bg = fg_features.shape[0], bg_features.shape[0]

    # Initialize DTW matrix
    dtw = np.full((n_fg + 1, n_bg + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0

    # Compute pairwise distances first (can be parallelized)
    distances = np.zeros((n_fg, n_bg), dtype=np.float64)
    for i in prange(n_fg):
        for j in range(n_bg):
            # Euclidean distance between feature vectors
            dist = 0.0
            for k in range(fg_features.shape[1]):
                diff = fg_features[i, k] - bg_features[j, k]
                dist += diff * diff
            distances[i, j] = np.sqrt(dist)

    # Normalize distances to [0, 1]
    max_dist = np.max(distances)
    if max_dist > 0:
        distances = distances / max_dist

    # Fill DTW matrix with Sakoe-Chiba band constraint
    for i in range(1, n_fg + 1):
        j_start = max(1, i - window)
        j_end = min(n_bg + 1, i + window)

        for j in range(j_start, j_end):
            cost = distances[i - 1, j - 1]

            # DTW recursion
            dtw[i, j] = cost + min(
                dtw[i - 1, j],  # Insertion
                dtw[i, j - 1],  # Deletion
                dtw[i - 1, j - 1],  # Match
            )

    return dtw


@jit(nopython=True, cache=True)
def find_dtw_path_numba(dtw_matrix: np.ndarray) -> np.ndarray:
    """
    Optimized DTW path finding using Numba.

    This replaces the backtracking loop in DTWAligner._find_optimal_path()
    with a JIT-compiled version.

    Args:
        dtw_matrix: Computed DTW cost matrix

    Returns:
        Optimal path as array of (i, j) pairs

    """
    n_fg, n_bg = dtw_matrix.shape
    n_fg -= 1
    n_bg -= 1

    # Use a pre-allocated array for path (worst case size)
    max_path_length = n_fg + n_bg
    path_array = np.zeros((max_path_length, 2), dtype=np.int32)
    path_idx = 0

    # Backtrack to find path
    i, j = n_fg, n_bg

    while i > 0 and j > 0:
        path_array[path_idx, 0] = i - 1  # Convert to 0-based indices
        path_array[path_idx, 1] = j - 1
        path_idx += 1

        # Choose direction with minimum cost
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            costs = np.array(
                [
                    dtw_matrix[i - 1, j],  # From above
                    dtw_matrix[i, j - 1],  # From left
                    dtw_matrix[i - 1, j - 1],  # From diagonal
                ]
            )

            min_idx = np.argmin(costs)
            if min_idx == 0:
                i -= 1
            elif min_idx == 1:
                j -= 1
            else:
                i -= 1
                j -= 1

    # Add remaining path
    while i > 0:
        i -= 1
        path_array[path_idx, 0] = i
        path_array[path_idx, 1] = 0
        path_idx += 1
    while j > 0:
        j -= 1
        path_array[path_idx, 0] = 0
        path_array[path_idx, 1] = j
        path_idx += 1

    # Trim and reverse
    path_array = path_array[:path_idx]
    return path_array[::-1]


# ==============================================================================
# Fingerprint Similarity Optimizations (3-10x speedup)
# ==============================================================================


@jit(nopython=True, parallel=True, cache=True)
def compute_hamming_distances_batch(
    hashes1: np.ndarray, hashes2: np.ndarray
) -> np.ndarray:
    """
    Batch computation of Hamming distances between hash arrays.

    This accelerates fingerprint comparisons when comparing many
    frames at once.

    Args:
        hashes1: First set of hashes (N, hash_size)
        hashes2: Second set of hashes (M, hash_size)

    Returns:
        Distance matrix (N, M)

    """
    n1, n2 = hashes1.shape[0], hashes2.shape[0]
    distances = np.zeros((n1, n2), dtype=np.float64)

    for i in prange(n1):
        for j in range(n2):
            # Hamming distance
            distance = 0.0
            for k in range(hashes1.shape[1]):
                if hashes1[i, k] != hashes2[j, k]:
                    distance += 1.0
            distances[i, j] = distance

    return distances


@jit(nopython=True, cache=True)
def compute_histogram_correlation(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Optimized histogram correlation computation.

    Replaces cv2.compareHist for faster execution when comparing
    many histograms.

    Args:
        hist1: First histogram
        hist2: Second histogram

    Returns:
        Correlation coefficient

    """
    # Normalize histograms
    sum1 = np.sum(hist1)
    sum2 = np.sum(hist2)

    if sum1 == 0 or sum2 == 0:
        return 0.0

    norm1 = hist1 / sum1
    norm2 = hist2 / sum2

    # Compute correlation
    mean1 = np.mean(norm1)
    mean2 = np.mean(norm2)

    numerator = 0.0
    denom1 = 0.0
    denom2 = 0.0

    for i in range(len(norm1)):
        diff1 = norm1[i] - mean1
        diff2 = norm2[i] - mean2
        numerator += diff1 * diff2
        denom1 += diff1 * diff1
        denom2 += diff2 * diff2

    denominator = np.sqrt(denom1 * denom2)
    if denominator == 0:
        return 0.0

    return numerator / denominator


@jit(nopython=True, cache=True)
def compute_weighted_similarity(
    hash_distances: np.ndarray, hist_correlation: float, weights: np.ndarray
) -> float:
    """
    Compute weighted similarity score from multiple hash distances.

    Args:
        hash_distances: Array of normalized hash distances
        hist_correlation: Histogram correlation score
        weights: Weight for each hash type

    Returns:
        Combined similarity score (0-1)

    """
    # Convert distances to similarities
    hash_similarities = 1.0 - hash_distances

    # Weighted sum
    weighted_sum = 0.0
    weight_sum = 0.0

    for i in range(len(hash_similarities)):
        weighted_sum += hash_similarities[i] * weights[i]
        weight_sum += weights[i]

    # Add histogram correlation with its weight
    if len(weights) > len(hash_similarities):
        weighted_sum += hist_correlation * weights[-1]
        weight_sum += weights[-1]

    if weight_sum == 0:
        return 0.0

    return weighted_sum / weight_sum


# ==============================================================================
# Helper Functions for Integration
# ==============================================================================


def prepare_fingerprints_for_numba(
    fingerprints: dict[int, dict[str, np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Convert fingerprint dictionary to numpy arrays for Numba processing.

    Args:
        fingerprints: Dictionary of frame fingerprints

    Returns:
        Tuple of (frame_indices, feature_matrix)

    """
    # Extract frame indices
    indices = np.array(sorted(fingerprints.keys()), dtype=np.int32)

    # Collect all hash types
    sample_fp = next(iter(fingerprints.values()))
    hash_types = [k for k in sample_fp.keys() if k != "histogram"]

    # Build feature matrix
    features = {}
    for hash_type in hash_types:
        hash_list = []
        for idx in indices:
            hash_data = fingerprints[idx][hash_type]
            hash_list.append(hash_data.flatten())
        features[hash_type] = np.array(hash_list)

    # Add histograms separately
    if "histogram" in sample_fp:
        hist_list = []
        for idx in indices:
            hist_list.append(fingerprints[idx]["histogram"])
        features["histogram"] = np.array(hist_list)

    return indices, features


def log_numba_compilation():
    """Log information about Numba JIT compilation."""
    logger.info("Numba JIT compilation in progress...")
    logger.info("First run will be slower due to compilation")
    logger.info("Subsequent runs will be significantly faster")
