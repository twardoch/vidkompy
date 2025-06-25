# Refactoring Plan for MVP Streamlining

## 🎯 CURRENT OBJECTIVES

Streamline the codebase for a performant, focused v1.0 MVP by identifying and implementing changes that reduce complexity, remove redundancy, and defer non-essential features, while maintaining core functionality.

## 📝 DETAILED PLAN

### Phase 1: Analysis and Planning (Completed)

1.  ✅ **Initial `PLAN.md` and `TODO.md` refinement.**
2.  ✅ **Codebase analysis for slimming opportunities.**

### Phase 2: MVP Streamlining Implementation Plan

#### A. Simplify `align` Module (Thumbnail Detection)

1.  **[P2.A1] Simplify `FeatureMatchingAlgorithm` (`align/algorithms.py`):**
    *   **What:** Modify to use only one primary feature detector (e.g., ORB or AKAZE) instead of SIFT, AKAZE, and ORB.
    *   **Why:** Reduce complexity and potential dependency issues (SIFT) for MVP. ORB/AKAZE are generally good balances.
    *   **How:**
        *   Remove SIFT and one of (AKAZE, ORB) from `_init_feature_detectors`.
        *   Simplify logic in `enhanced_feature_matching` to not switch between detectors.
        *   Update `detector_type` parameter if it becomes fixed.
    *   **Impact:** Simpler algorithm, potentially fewer dependencies if SIFT was an issue.

2.  **[P2.A2] Defer Advanced Alignment Algorithms (`align/algorithms.py`):**
    *   **What:** Conditionally disable or comment out `SubPixelRefinementAlgorithm`, `PhaseCorrelationAlgorithm`, and `HybridMatchingAlgorithm`.
    *   **Why:** These offer higher precision but add complexity. For MVP, `TemplateMatchingAlgorithm` and a simplified `FeatureMatchingAlgorithm` might suffice.
    *   **How:**
        *   In `align/precision.py`, adjust `_strategy_map` for higher precision levels (3 and 4) to perhaps reuse Level 2's strategy or a simpler refinement, effectively deferring the complex algorithms.
        *   If `PhaseCorrelationAlgorithm` is deferred, `scikit-image` might become an optional dependency.
    *   **Impact:** Reduced code paths, potentially fewer dependencies.

3.  **[P2.A3] Simplify `PrecisionAnalyzer` (`align/precision.py`):**
    *   **What:** Reduce the number of active precision levels for MVP (e.g., keep Levels 0, 1, and 2).
    *   **Why:** Fewer levels mean simpler logic and faster determination of a "good enough" match for MVP.
    *   **How:**
        *   Modify `progressive_analysis` to cap at `max_precision = 2` (or a configurable MVP cap).
        *   Adjust `_strategy_map` so levels 3 and 4 map to level 2's analysis or a simple pass-through of level 2's result.
        *   `SCALE_PARAMS` for deferred levels would not be actively used.
    *   **Impact:** Simpler control flow, faster alignment analysis.

#### B. Streamline `comp` Module (Video Composition)

1.  **[P2.B1] Consolidate Temporal Alignment Strategy:**
    *   **What:** Confirm that `TemporalSyncer` with its `TunnelFullSyncer` / `TunnelMaskSyncer` is the primary temporal alignment path for `vidkompy comp`. Defer or remove the more complex `PreciseTemporalAlignment` and `MultiResolutionAligner` path for MVP.
    *   **Why:** The README and current code structure suggest Tunnel syncers are the main modern approach. Focusing on one primary strategy simplifies the `comp` module significantly. DTW-based approaches (`DTWSyncer`, `MultiResolutionAligner`) are powerful but add considerable complexity.
    *   **How:**
        *   If confirmed, comment out or remove usages and initialization of `PreciseTemporalAlignment` and `MultiResolutionAligner` within the main composition flow.
        *   This would also mean `DTWSyncer` might only be needed if the `align` module's higher precision levels (which might use it indirectly) are kept. If those are also simplified, `DTWSyncer` could be deferred.
        *   Relevant Numba ops in `utils/numba_ops.py` (e.g., `apply_polynomial_drift_correction`, DTW helpers) might also be deferred.
    *   **Impact:** Major simplification of the temporal alignment pipeline, making `TemporalSyncer` and its Tunnel engines the core.

2.  **[P2.B2] Simplify `TemporalSyncer` Engine Choice (`comp/temporal.py`):**
    *   **What:** For MVP, consider making `TunnelFullSyncer` the default and only temporal alignment engine.
    *   **Why:** `TunnelFullSyncer` is described as the fastest with perfect confidence for standard videos. The `TunnelMaskSyncer` (for letterboxed content) could be a post-MVP enhancement if it adds significant maintenance overhead.
    *   **How:**
        *   Modify `TemporalSyncer` to default to `TunnelFullSyncer`.
        *   Remove the `engine` parameter from `composite_videos` CLI and `AlignmentEngine` if only one option remains for MVP.
        *   If `TunnelMaskSyncer` is deferred, `compute_masked_fingerprint` in `FrameFingerprinter` might also be deferred.
    *   **Impact:** Simpler CLI, more focused temporal alignment.

3.  **[P2.B3] Review `TimeMode.BORDER` (`comp/align.py`, `utils/enums.py`):**
    *   **What:** Investigate the usage and necessity of `TimeMode.BORDER`. It seems distinct from the `mask` engine in `TemporalSyncer`.
    *   **Why:** If it's a legacy or rarely used feature adding complexity (e.g. `create_border_mask`), it could be deferred for MVP.
    *   **How:**
        *   Trace its usage. If not essential, remove related logic and simplify `TimeMode` enum.
    *   **Impact:** Reduced code in `AlignmentEngine` and `TemporalSyncer`.

4.  **[P2.B4] Simplify `FrameFingerprinter` (`comp/fingerprint.py`):**
    *   **What:** Reduce the number of hash algorithms used (e.g., to PHash and ColorMomentHash, or just PHash + histogram).
    *   **Why:** Fewer hashes simplify computation, comparison logic, and the `weight_map`.
    *   **How:**
        *   Modify `_init_hashers` to initialize fewer hashers.
        *   Update `compute_fingerprint` and `compare_fingerprints` accordingly.
    *   **Impact:** Faster fingerprinting, simpler comparison logic.

#### C. General Codebase and Utility Refinements

1.  **[P2.C1] Fix Typos and Minor Issues:**
    *   **What:** Correct typo in `.github/workflows/push.yml` (`vidkompo` to `vidkompy`).
    *   **Why:** Ensure CI works correctly.
    *   **How:** Edit the file.
    *   **Impact:** CI reliability.

2.  **[P2.C2] Review Dependencies (`pyproject.toml`):**
    *   **What:** Check if `soundfile` is used. If `PhaseCorrelationAlgorithm` is deferred, check if `scikit-image` is still a hard dependency.
    *   **Why:** Minimize dependency footprint for MVP.
    *   **How:** Analyze import graph after other simplifications.
    *   **Impact:** Potentially smaller environment.

3.  **[P2.C3] Update `benchmark.sh`:**
    *   **What:** Ensure `benchmark.sh` aligns with any changes to CLI parameters or available engines.
    *   **Why:** Keep benchmark script relevant.
    *   **How:** Edit the script after other refactorings.
    *   **Impact:** Accurate benchmarking.

4.  **[P2.C4] Review `package.toml`:**
    *   **What:** Determine if `package.toml` is actively used by the current build/packaging setup (Hatch).
    *   **Why:** Remove if obsolete.
    *   **How:** Check Hatch configuration and build process.
    *   **Impact:** Cleaner project root if unused. (Low priority)

### Phase 3: Implementation and Testing

*For each item implemented from Phase 2:*

1.  **Implement the change.**
    *   Write or modify code.
    *   Write tests if new logic is introduced or significantly altered (focus on maintaining coverage for MVP features).
2.  **Test the change.**
    *   Run relevant unit tests.
    *   Perform manual testing for core composition and alignment tasks.
    *   Run `./cleanup.sh`.
3.  **Update `CHANGELOG.md`** with a description of the change.
4.  **Mark the corresponding item in `TODO.md` and `PLAN.md` as complete.**

### Phase 4: Final Review and Submission

1.  **Holistic Review:**
    *   Ensure all changes are coherent and the codebase is in a stable state for MVP.
    *   Verify that core functionality (e.g., `vidkompy comp` with default settings, `vidkompy align` with simplified precision) works well.
2.  **Final Testing:**
    *   Run all tests using `./cleanup.sh`.
3.  **Update `README.md`:**
    *   Reflect any significant changes to functionality, usage, or simplified algorithm descriptions.
4.  **Submit the changes** with a comprehensive commit message.

This updated plan provides specific, actionable items for streamlining the codebase towards an MVP.Next, `TODO.md`:
