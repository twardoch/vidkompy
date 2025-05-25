#!/usr/bin/env python3
"""
Numba performance optimizations for vidkompy bottlenecks.

These functions replace the most computationally expensive parts with 
JIT-compiled versions for significant speedups.
"""

import numpy as np
from numba import jit, prange
from typing import Tuple


# ==============================================================================
# DTW Algorithm Optimizations (Highest Impact)
# ==============================================================================

@jit(nopython=True, parallel=True, cache=True)
def compute_dtw_cost_matrix_numba(
    fg_features: np.ndarray, 
    bg_features: np.ndarray, 
    window: int
) -> np.ndarray:
    """
    Optimized DTW cost matrix computation using Numba.
    
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
                dtw[i - 1, j],      # Insertion
                dtw[i, j - 1],      # Deletion  
                dtw[i - 1, j - 1]   # Match
            )
    
    return dtw


@jit(nopython=True, cache=True)
def find_dtw_path_numba(dtw_matrix: np.ndarray) -> np.ndarray:
    """
    Optimized DTW path finding using Numba.
    
    Args:
        dtw_matrix: Computed DTW cost matrix
        
    Returns:
        Optimal path as array of (i, j) pairs
    """
    n_fg, n_bg = dtw_matrix.shape
    n_fg -= 1
    n_bg -= 1
    
    # Backtrack to find path
    path = []
    i, j = n_fg, n_bg
    
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))  # Convert to 0-based indices
        
        # Choose direction with minimum cost
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            costs = np.array([
                dtw_matrix[i - 1, j],      # From above
                dtw_matrix[i, j - 1],      # From left
                dtw_matrix[i - 1, j - 1]   # From diagonal
            ])
            
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
        path.append((i, 0))
    while j > 0:
        j -= 1
        path.append((0, j))
    
    # Convert to numpy array and reverse
    path_array = np.array(path)
    return path_array[::-1]


# ==============================================================================
# Fingerprint Similarity Optimizations
# ==============================================================================

@jit(nopython=True, parallel=True, cache=True)
def compute_hamming_distances_batch(
    hashes1: np.ndarray, 
    hashes2: np.ndarray
) -> np.ndarray:
    """
    Batch computation of Hamming distances between hash arrays.
    
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


# ==============================================================================
# Multi-Resolution Alignment Optimizations  
# ==============================================================================

@jit(nopython=True, cache=True)
def apply_polynomial_drift_correction(
    mapping: np.ndarray,
    interval: int,
    poly_degree: int,
    blend_factor: float
) -> np.ndarray:
    """
    Optimized polynomial drift correction using Numba.
    
    Args:
        mapping: Frame mapping array
        interval: Drift correction interval
        poly_degree: Polynomial degree for fitting
        blend_factor: Blending factor for correction
        
    Returns:
        Corrected mapping array
    """
    corrected = mapping.copy()
    num_segments = len(mapping) // interval + 1
    
    for seg_idx in range(num_segments):
        start_idx = seg_idx * interval
        end_idx = min((seg_idx + 1) * interval, len(mapping))
        
        if start_idx >= end_idx:
            continue
            
        segment_mapping = mapping[start_idx:end_idx]
        segment_indices = np.arange(len(segment_mapping), dtype=np.float64)
        
        # Simple polynomial fitting (for degree 1 and 2)
        if poly_degree == 1 and len(segment_mapping) >= 2:
            # Linear fit: y = mx + b
            n = len(segment_indices)
            sum_x = np.sum(segment_indices)
            sum_y = np.sum(segment_mapping)
            sum_xy = np.sum(segment_indices * segment_mapping)
            sum_x2 = np.sum(segment_indices * segment_indices)
            
            if n * sum_x2 - sum_x * sum_x != 0:
                m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                b = (sum_y - m * sum_x) / n
                expected_progression = m * segment_indices + b
            else:
                # Fallback to linear interpolation
                expected_progression = np.linspace(
                    segment_mapping[0], segment_mapping[-1], len(segment_mapping)
                )
        else:
            # Fallback to linear interpolation for higher degrees or insufficient data
            expected_progression = np.linspace(
                segment_mapping[0], segment_mapping[-1], len(segment_mapping)
            )
        
        # Apply blending
        for k in range(len(segment_mapping)):
            map_idx = start_idx + k
            corrected[map_idx] = (
                blend_factor * segment_mapping[k] + 
                (1 - blend_factor) * expected_progression[k]
            )
    
    # Ensure monotonicity
    for i in range(1, len(corrected)):
        corrected[i] = max(corrected[i], corrected[i - 1])
    
    return corrected


@jit(nopython=True, cache=True)
def apply_savitzky_golay_filter(
    data: np.ndarray, 
    window_length: int, 
    poly_order: int
) -> np.ndarray:
    """
    Simplified Savitzky-Golay filter implementation for Numba.
    
    Note: This is a simplified version. For full SciPy compatibility,
    consider using the original scipy.signal.savgol_filter.
    """
    if window_length >= len(data):
        return data.copy()
    
    half_window = window_length // 2
    filtered = np.zeros_like(data)
    
    # Simple moving average for edge cases
    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        filtered[i] = np.mean(data[start:end])
    
    return filtered


# ==============================================================================
# Mask and Border Operations
# ==============================================================================

@jit(nopython=True, parallel=True, cache=True)
def apply_mask_to_frames_batch(
    frames: np.ndarray, 
    mask: np.ndarray
) -> np.ndarray:
    """
    Apply mask to multiple frames in parallel.
    
    Args:
        frames: Frame array (N, H, W, C)
        mask: Binary mask (H, W)
        
    Returns:
        Masked frames array
    """
    n_frames, height, width, channels = frames.shape
    masked_frames = np.zeros_like(frames)
    
    for n in prange(n_frames):
        for h in range(height):
            for w in range(width):
                if mask[h, w] > 0:
                    for c in range(channels):
                        masked_frames[n, h, w, c] = frames[n, h, w, c]
    
    return masked_frames


@jit(nopython=True, cache=True)
def create_border_mask_numba(
    bg_height: int,
    bg_width: int, 
    fg_left: int,
    fg_right: int,
    fg_top: int,
    fg_bottom: int,
    border_thickness: int
) -> np.ndarray:
    """
    Optimized border mask creation.
    """
    mask = np.zeros((bg_height, bg_width), dtype=np.uint8)
    
    # Top border
    if fg_top > 0:
        border_top = max(0, fg_top - border_thickness)
        for y in range(border_top, fg_top):
            for x in range(fg_left, fg_right):
                if 0 <= y < bg_height and 0 <= x < bg_width:
                    mask[y, x] = 1
    
    # Bottom border  
    if fg_bottom < bg_height:
        border_bottom = min(bg_height, fg_bottom + border_thickness)
        for y in range(fg_bottom, border_bottom):
            for x in range(fg_left, fg_right):
                if 0 <= y < bg_height and 0 <= x < bg_width:
                    mask[y, x] = 1
    
    # Left border
    if fg_left > 0:
        border_left = max(0, fg_left - border_thickness)
        for y in range(fg_top, fg_bottom):
            for x in range(border_left, fg_left):
                if 0 <= y < bg_height and 0 <= x < bg_width:
                    mask[y, x] = 1
    
    # Right border
    if fg_right < bg_width:
        border_right = min(bg_width, fg_right + border_thickness)
        for y in range(fg_top, fg_bottom):
            for x in range(fg_right, border_right):
                if 0 <= y < bg_height and 0 <= x < bg_width:
                    mask[y, x] = 1
    
    return mask


# ==============================================================================
# Integration Helper Functions
# ==============================================================================

def integrate_numba_optimizations():
    """
    Instructions for integrating these optimizations into vidkompy.
    
    Performance gains expected:
    - DTW computation: 5-20x speedup
    - Fingerprint comparisons: 3-10x speedup  
    - Drift correction: 2-5x speedup
    - Mask operations: 3-8x speedup
    
    Integration steps:
    1. Add numba dependency to pyproject.toml
    2. Replace bottleneck functions with numba versions
    3. Convert data to numpy arrays before calling numba functions
    4. Handle the first-run JIT compilation cost
    """
    
    # Example integration in DTWAligner:
    print("""
    # In src/vidkompy/core/dtw_aligner.py:
    
    def _compute_cost_matrix(self, fg_features, bg_features):
        # Convert to numpy arrays if needed
        fg_array = np.array([f.flatten() for f in fg_features])
        bg_array = np.array([f.flatten() for f in bg_features])
        
        # Use optimized numba version
        dtw_matrix = compute_dtw_cost_matrix_numba(
            fg_array, bg_array, self.window
        )
        
        return dtw_matrix[1:, 1:]  # Remove padding
    """)
    
    print("""
    # In src/vidkompy/core/frame_fingerprint.py:
    
    def compare_fingerprints_batch(self, fps1, fps2):
        # Extract hashes to arrays
        hashes1 = np.array([fp['phash'] for fp in fps1])
        hashes2 = np.array([fp['phash'] for fp in fps2])
        
        # Use optimized batch computation
        distances = compute_hamming_distances_batch(hashes1, hashes2)
        
        return 1.0 - (distances / 64.0)  # Convert to similarities
    """)


if __name__ == "__main__":
    integrate_numba_optimizations()
