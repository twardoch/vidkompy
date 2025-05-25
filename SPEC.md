# Vidkompy Precise Temporal Alignment Engine: Analysis and Improvement Plan

## 1. Current Precise Engine Workflow

The `precise` temporal alignment engine in `vidkompy` aims to achieve highly accurate frame-to-frame mapping between a foreground (FG) video and a background (BG) video. It's a multi-stage process primarily orchestrated by `PreciseTemporalAlignment` which utilizes `MultiResolutionAligner` and `DTWAligner`.

The input BG frames are spatially cropped to the FG's region before temporal alignment begins.

The core workflow is as follows:

**Phase 0: Fingerprint Computation** (`PreciseTemporalAlignment.align`)
-   Frame fingerprints (compact visual representations) are computed for all frames of both FG and BG videos using `FrameFingerprinter.compute_fingerprints`. These are numpy arrays of feature vectors.

**Phase 1: Multi-Resolution Alignment** (`MultiResolutionAligner.align`)
This phase performs a coarse-to-fine alignment.
1.  **Temporal Pyramid Creation** (`create_temporal_pyramid`):
    *   Fingerprint arrays for FG and BG are downsampled at multiple rates (e.g., 1/16, 1/8, 1/4, 1/2 of total frames) creating a "pyramid" of lower-resolution fingerprint sequences.
2.  **Coarse Alignment** (`coarse_alignment`):
    *   DTW (Dynamic Time Warping) is performed using `DTWAligner` on the fingerprints from the coarsest resolution level of both pyramids. This yields an initial, rough mapping of FG frames to BG frames. The DTW window here is relative to the length of the BG samples.
3.  **Hierarchical Refinement** (`hierarchical_alignment` calling `refine_alignment`):
    *   The alignment is iteratively refined by moving from coarser to finer resolution levels in the pyramid.
    *   In each `refine_alignment` step:
        *   The mapping from the previous (coarser) level is scaled up to the current (finer) resolution.
        *   For segments of FG frames at the current resolution, a local DTW alignment is performed. This local DTW searches for matches in a BG frame segment defined by the scaled-up coarse mapping plus/minus a `refinement_window` (e.g., 30 frames). The DTW window for this local alignment is small (e.g., 10 frames or half the BG segment length).
4.  **Drift Correction** (`apply_drift_correction`):
    *   The mapping obtained from the hierarchical alignment (at the finest pyramid level, still sparse) is corrected for drift.
    *   The mapping is divided into segments based on `drift_correction_interval` (e.g., every 100 frames).
    *   For each segment, a linear progression is assumed between its start and end points.
    *   The original mapping within the segment is blended with this linear progression using a `drift_blend_factor` (e.g., 0.85, meaning 85% trust in original mapping, 15% in linear).
    *   The corrected mapping is forced to be monotonic.
5.  **Interpolation to Full Mapping** (`interpolate_full_mapping`):
    *   The corrected sparse mapping (from the finest pyramid resolution) is linearly interpolated to produce a frame-by-frame mapping for every FG frame. This ensures monotonicity.
    *   This interpolated mapping and a confidence score are returned by the `MultiResolutionAligner`.

**Phase 2: Keyframe Detection and Alignment** (`PreciseTemporalAlignment.align`)
1.  **Keyframe Detection** (`detect_keyframes`):
    *   Keyframes are detected independently in FG and BG videos. This involves:
        *   Calculating temporal differences between consecutive frame fingerprints.
        *   Smoothing these differences using a Gaussian filter.
        *   Finding peaks in the smoothed differences (representing scene changes or significant motion). First and last frames are always included.
2.  **Keyframe Alignment** (`align_keyframes`):
    *   The detected FG keyframes are aligned to the detected BG keyframes using `DTWAligner` on their respective fingerprints. The DTW window here is the number of BG keyframes. This produces a sparse mapping between specific FG keyframe indices and BG keyframe indices.

**Phase 3: Bidirectional DTW Refinement** (`PreciseTemporalAlignment.align`)
1.  **Sampling**: FG and BG fingerprints are sampled (e.g., max 500 samples each).
2.  **Bidirectional DTW** (`bidirectional_dtw`):
    *   Forward DTW: Aligns sampled FG fingerprints to sampled BG fingerprints.
    *   Backward DTW: Aligns sampled BG fingerprints (reversed) to sampled FG fingerprints (reversed). The resulting path is then reversed again.
    *   The two mappings (forward and backward) are averaged.
    *   The averaged mapping is forced to be monotonic.
3.  **Interpolation**: The sparse averaged mapping is linearly interpolated to full FG frame resolution.

**Phase 4: Combination and Final Refinements** (`PreciseTemporalAlignment.align`)
1.  **Combine Mappings**:
    *   The full mapping from Phase 1 (Multi-Resolution) and the full mapping from Phase 3 (Bidirectional DTW) are combined using a weighted average (currently 50/50).
2.  **Apply Keyframe Constraints**:
    *   The specific keyframe-to-keyframe alignments from Phase 2 are enforced onto the combined mapping.
    *   Linear interpolation is performed on the combined mapping *between* these enforced keyframe anchor points.
3.  **Sliding Window Refinement** (`refine_with_sliding_window`):
    *   The current mapping is further refined using a sliding window approach.
    *   For segments (windows) of FG frames (e.g., 30 frames):
        *   A search range is determined in the BG frames around the current mapping for that segment.
        *   The algorithm attempts to find the best offset within this search range by minimizing the sum of fingerprint distances between the FG window and the corresponding BG window.
    *   The resulting refined mapping is smoothed using a Gaussian filter.
    *   Monotonicity is enforced.
4.  **Finalization**:
    *   The final mapping is clipped to ensure all BG frame indices are within the valid range of the BG video.
    *   An overall confidence score for the alignment is computed.

## 2. Identified Issues

1.  **"Flag Wave" Drifts**: The primary issue reported is that the `precise` engine exhibits "flag wave drifts," where the background video appears to speed up and slow down inconsistently relative to the foreground. This indicates that the temporal mapping is not smooth or stable.
2.  **Complexity and Potential Interactions**: The multi-phase approach with several stages of alignment, interpolation, averaging, and refinement is highly complex. It's possible that these stages interact in unintended ways, leading to oscillations or over-corrections.
3.  **Drift Correction Mechanism**: The `apply_drift_correction` in `MultiResolutionAligner` uses a simple linear interpolation as a baseline for correction. If the actual motion over the `drift_correction_interval` is non-linear, this can introduce errors. The `drift_blend_factor` might also be a point of sensitivity.
4.  **Effectiveness of CLI Parameters**:
    *   The `-d` (drift_interval) parameter seems to be correctly passed to the `MultiResolutionAligner`.
    *   The `-w` (window) parameter for DTW is **not effectively used** by the `precise` engine. Most internal calls to `DTWAligner` instantiate it with hardcoded or dynamically calculated window sizes, overriding the CLI-provided value. This is a bug.
        *   `MultiResolutionAligner.coarse_alignment`: Uses `initial_window_ratio`.
        *   `MultiResolutionAligner.refine_alignment`: Uses a small, hardcoded window (e.g., 10).
        *   `PreciseTemporalAlignment.align_keyframes`: Instantiates `DTWAligner` with `window=len(bg_keyframes)`.
        *   `PreciseTemporalAlignment.bidirectional_dtw`: Instantiates `DTWAligner` with `window=50`.
5.  **Averaging and Interpolation**: Multiple averaging and interpolation steps might smooth out true variations or introduce their own artifacts if not handled carefully.

## 3. Proposed Improvements to Precise Engine

### 3.1. Idea 1: Enhanced Drift Correction & Smoothing in Multi-Resolution Aligner (Targeted Fix)

*   **Problem Addressed**: Directly targets the "flag wave" drift potentially originating from or exacerbated by the current drift correction and lack of sufficient smoothing in the multi-resolution phase.
*   **Proposal**:
    1.  **Improve Baseline for Drift Correction**: In `MultiResolutionAligner.apply_drift_correction`, replace the simple linear interpolation baseline (between segment start and end) with a more robust local regression model (e.g., LOESS - Locally Estimated Scatterplot Smoothing, or a polynomial fit) applied to the original mapping *within* each `drift_correction_interval` segment. The `drift_blend_factor` would then blend the original mapping with this more sophisticated local trend.
    2.  **Adaptive Blend Factor**: Make `drift_blend_factor` adaptive. If the original mapping within a segment is already very smooth or consistent (low variance in frame-to-frame deltas), use a higher blend factor (trust original more). If it's erratic, use a lower blend factor (trust the smoothed baseline more).
    3.  **Global Smoothing Pass**: After the `hierarchical_alignment` and `apply_drift_correction` steps in `MultiResolutionAligner.align`, and *before* `interpolate_full_mapping`, apply a global smoothing filter (e.g., Savitzky-Golay filter) to the `corrected_mapping`. This filter is good at preserving the overall shape of the curve while removing high-frequency noise (which could manifest as "flag waves"). The window size and polynomial order for Savitzky-Golay would need tuning.
    4.  **Review Interpolation**: Ensure that `interpolate_full_mapping` uses a method that preserves smoothness (e.g., spline interpolation if linear is too coarse, though linear with enough points from Savitzky-Golay might be fine).

### 3.2. Idea 2: Optical Flow-Assisted Consistency Check & Refinement (Motion-Aware Fix)

*   **Problem Addressed**: "Flag wave" drift implies unnatural accelerations/decelerations in the background. Fingerprint similarity alone might not capture this. Optical flow can help assess motion consistency.
*   **Proposal**:
    1.  **Initial Global Alignment**: Obtain a primary global frame mapping (e.g., from the improved Multi-Resolution Aligner as per Idea 1, or a simplified DTW pass).
    2.  **Motion Consistency Verification**: For the mapped BG frames `(bg_i, bg_i+1, bg_i+2, ...)` corresponding to sequential FG frames `(fg_i, fg_i+1, fg_i+2, ...)`:
        *   Calculate optical flow between `bg_frame[bg_i]` and `bg_frame[bg_i+1]`.
        *   Use this flow to predict `bg_frame[bg_i+2_predicted]` from `bg_frame[bg_i+1]`.
        *   Compare `bg_frame[bg_i+2_predicted]` with the actual `bg_frame[bg_i+2]` chosen by the mapping. A large discrepancy suggests inconsistent motion.
    3.  **Refinement based on Flow**:
        *   Identify segments in the mapping where implied BG motion is inconsistent (high optical flow prediction error).
        *   For these segments, re-evaluate the BG frame choices by adding a penalty to the DTW cost or similarity score for mappings that result in high optical flow prediction errors. This would favor BG frame sequences that imply smoother, more natural motion.
    *   This could be a new refinement pass after Phase 4, or integrated into the sliding window refinement logic.

### 3.3. Idea 3: Dominant Path DTW with Iterative Refinement (Simplification & Robustness)

*   **Problem Addressed**: The current engine's complexity might be counterproductive. A simpler, more robust core alignment followed by targeted refinement could be better.
*   **Proposal**:
    1.  **Primary DTW Alignment**:
        *   Perform a single, robust DTW alignment (`DTWAligner`) on moderately sampled fingerprints (e.g., every 3-5 frames for both FG and BG, ensuring enough detail but manageable cost matrix size). Use the actual CLI `-w` parameter for the Sakoe-Chiba band, fixing the current bug.
        *   This produces a primary sparse, monotonic mapping.
    2.  **Segment Quality Analysis**:
        *   Analyze this primary mapping to identify "low-quality" segments based on:
            *   High accumulated DTW cost within the segment.
            *   Steep changes in the mapping slope (indicating rapid, potentially unnatural speed changes).
            *   Low average fingerprint similarity for matches in the segment.
    3.  **Targeted Re-alignment of Problem Segments**:
        *   For each identified low-quality segment:
            *   Extract FG and BG frames *only for that segment* at a higher density (e.g., all frames).
            *   Perform a local DTW alignment for just this segment, possibly with a more refined similarity measure (e.g., if hashes were used initially, try SSIM or a combination). The DTW search can be constrained by the endpoints of the segment from the primary mapping.
    4.  **Stitching and Smoothing**:
        *   Replace the low-quality segments in the primary mapping with their re-aligned versions, ensuring overall monotonicity is maintained at stitch points.
        *   Interpolate the resulting (still potentially sparse) mapping to full FG frame resolution.
        *   Apply a final global smoothing filter (e.g., Savitzky-Golay) to the entire full-resolution mapping.
    *   This approach focuses computational effort where it's most needed and reduces the number of intermediate global mappings and complex interactions between multiple alignment strategies.

## 4. Detailed Specification for Improvement Idea 1

**Improvement Goal**: Enhance the `MultiResolutionAligner` to produce a smoother and more drift-resistant sparse mapping by improving its internal drift correction and adding a global smoothing pass.

**Target Module**: `src/vidkompy/core/multi_resolution_aligner.py`

**Affected Methods**:
*   `MultiResolutionAligner.apply_drift_correction`
*   `MultiResolutionAligner.align` (to add the global smoothing pass)
*   Potentially `MultiResolutionAligner.interpolate_full_mapping` (if interpolation method needs change)

**Parameter Changes (Consider adding to `PreciseEngineConfig`)**:
*   `drift_correction_model`: str, e.g., "linear" (current), "polynomial", "loess" (default: "polynomial").
*   `poly_degree`: int, degree for polynomial regression (if "polynomial" model chosen, default: 2 or 3).
*   `loess_frac`: float, fraction of data for LOESS (if "loess" model chosen, default: 0.25).
*   `adaptive_blend_factor`: bool, whether to make `drift_blend_factor` adaptive (default: True).
*   `savitzky_golay_window`: int, window size for Savitzky-Golay filter (odd number, e.g., 11 or 21). Must be smaller than the number of data points.
*   `savitzky_golay_polyorder`: int, polynomial order for Savitzky-Golay filter (e.g., 2 or 3). Must be less than `savitzky_golay_window`.
*   `interpolation_method`: str, e.g., "linear" (current), "spline" (default: "linear").

**Step-by-Step Implementation Plan**:

**A. Modify `MultiResolutionAligner.apply_drift_correction`:**

1.  **Parameter Integration**:
    *   Update `PreciseEngineConfig` to include new parameters: `drift_correction_model`, `poly_degree`, `loess_frac`, `adaptive_blend_factor`.
    *   The `apply_drift_correction` method should access these from `self.config`.
2.  **Implement Alternative Baseline Models**:
    *   Inside the loop over `seg` (segments):
        *   Let `segment_mapping = mapping[start:end]`.
        *   Let `segment_indices = np.arange(len(segment_mapping))`.
        *   If `self.config.drift_correction_model == "polynomial"`:
            *   Fit a polynomial of degree `self.config.poly_degree` to `segment_mapping` vs `segment_indices`.
            *   `expected_segment_progression = polynomial_predict(segment_indices)`.
            *   Requires `numpy.polyfit` and `numpy.poly1d`.
        *   Else if `self.config.drift_correction_model == "loess"`:
            *   (Requires `statsmodels` or a custom implementation if heavy dependencies are to be avoided. If `statsmodels` is too heavy, stick to polynomial or keep linear for now and note dependency).
            *   `expected_segment_progression = loess_smooth(segment_mapping, frac=self.config.loess_frac)`.
        *   Else (current "linear" model):
            *   `expected_start = mapping[start]`
            *   `expected_end = mapping[min(end, len(mapping) - 1)]` (or rather, `segment_mapping[-1]`)
            *   `expected_segment_progression = np.linspace(expected_start, expected_end, len(segment_mapping))`
3.  **Implement Adaptive Blend Factor**:
    *   If `self.config.adaptive_blend_factor` is True:
        *   Calculate variance or standard deviation of `np.diff(segment_mapping)`.
        *   Normalize this variance/std (e.g., divide by mean delta-T or a typical value).
        *   Compute `current_blend_factor = self.config.drift_blend_factor + (some_scale * (1 - normalized_variance))`. Clamp between, say, 0.5 and 0.95.
    *   Else:
        *   `current_blend_factor = self.config.drift_blend_factor`.
4.  **Apply Blending**:
    *   Loop `i` from `start` to `end` (or `k` from `0` to `len(segment_mapping)-1`):
        *   `blend_factor_to_use = current_blend_factor` (if adaptive) or `self.config.drift_blend_factor`.
        *   `corrected[i] = blend_factor_to_use * mapping[i] + (1 - blend_factor_to_use) * expected_segment_progression[k]`.

**B. Modify `MultiResolutionAligner.align` for Global Smoothing**:

1.  **Parameter Integration**:
    *   Update `PreciseEngineConfig` for `savitzky_golay_window` and `savitzky_golay_polyorder`.
2.  **Add Smoothing Step**:
    *   After the line `corrected_mapping = self.apply_drift_correction(sparse_mapping)`:
    *   Check if `len(corrected_mapping) > self.config.savitzky_golay_window`. If not, skip smoothing or warn.
    *   `smoothed_mapping = scipy.signal.savgol_filter(corrected_mapping, window_length=self.config.savitzky_golay_window, polyorder=self.config.savitzky_golay_polyorder)`.
    *   Ensure `smoothed_mapping` is integer type and monotonic (force monotonicity if filter breaks it slightly: `for i in range(1, len(smoothed_mapping)): smoothed_mapping[i] = max(smoothed_mapping[i], smoothed_mapping[i-1])`).
    *   Pass `smoothed_mapping` (instead of `corrected_mapping`) to `self.interpolate_full_mapping`.
    *   Requires `scipy.signal.savgol_filter`. Add `scipy` to dependencies if not already there for this specific function (it is, based on `pyproject.toml`).

**C. Review `MultiResolutionAligner.interpolate_full_mapping`**:

1.  **Parameter Integration**:
    *   Update `PreciseEngineConfig` for `interpolation_method`.
2.  **Implement Spline Interpolation (Optional)**:
    *   If `self.config.interpolation_method == "spline"`:
        *   The input `sparse_mapping` comes from `source_resolution` steps.
        *   Create `x_sparse = np.arange(len(sparse_mapping)) * source_resolution`.
        *   Create `y_sparse = sparse_mapping`.
        *   Create `x_full = np.arange(target_length)`.
        *   Use `scipy.interpolate.CubicSpline` or `interp1d` with `kind='cubic'` to interpolate `y_sparse` at `x_full`.
        *   `full_mapping = spline_interpolator(x_full).astype(int)`.
        *   Ensure monotonicity.
    *   Current linear interpolation should be fine if the input `sparse_mapping` is already well-smoothed by Savitzky-Golay. Default to linear.

**D. Testing and Tuning**:

1.  **Unit Tests**: Add unit tests for `apply_drift_correction` with different models, and for the Savitzky-Golay smoothing step.
2.  **Integration Testing**: Test the precise engine with videos known to cause "flag wave" drift.
3.  **Parameter Tuning**: Experiment with default values for the new config parameters (`poly_degree`, `savitzky_golay_window`, `savitzky_golay_polyorder`) to find a good balance. The benchmark script from `TODO.md` can be adapted for this.
4.  **CLI Parameter Bug**: Separately, address the `-w` (window) CLI parameter not being used correctly in `PreciseTemporalAlignment` by ensuring the `DTWAligner` instances use the `self.dtw_aligner` (which is configured by `TemporalAligner` from the CLI param) or correctly pass the window parameter through.

**Rationale for Chosen Parameters**:
*   Polynomial/LOESS for drift correction offers more flexibility than simple linear interpolation to capture local trends.
*   Adaptive blend factor allows the correction to be gentle where the mapping is already stable and more assertive where it's erratic.
*   Savitzky-Golay filter is chosen for its ability to smooth data while preserving the general shape of the signal, which is crucial for temporal mappings.
*   Spline interpolation (optional) can provide smoother transitions than linear if the sparse points are far apart.

This plan aims to make the multi-resolution alignment phase more robust against local oscillations and produce a smoother input for the subsequent combination and refinement stages in `PreciseTemporalAlignment`.
