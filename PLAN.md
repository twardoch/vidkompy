# Refactoring Plan

Below is the **remaining refactoring work**. Major improvements have been completed including utils module refactoring, precision analysis improvements, domain model enhancements, and enum cleanup.

H2 sections list files; H3 subsections drill down to concrete objects that deserve work.
Actions are phrased imperatively so they read like commit messages.

---

## ✅ Completed

- **utils module refactoring**: Improved logging, correlation, and __all__ exports
- **precision analysis**: Replaced lambdas with namedtuple, added cached_property
- **domain models**: Renamed SpatialAlignment → SpatialTransform, added AudioInfo, from_path() method  
- **enum cleanup**: Removed FRAMES alias, updated call sites
- **type safety**: Added NotImplementedError for unknown algorithms

---

## 🔄 Remaining Work

## src/vidkompy/align/algorithms.py

### High Priority Improvements

* **Split `enhanced_feature_matching()`**: Break into `_choose_detector()`, `_match_descriptors()`, `_estimate_transform()` helpers (currently 79+ statements, too complex)
* **Replace bare `except`**: Surface unexpected errors in verbose mode instead of silent failure
* **Factory pattern for detectors**: Replace `if/elif` chain with enum-to-cv2-factory map

### Medium Priority Optimizations

* **Extract scale-grid generation**: Move to `_iter_scales()` to remove duplication
* **Early return optimization**: Return early on invalid scale instead of constructing zero-confidence `MatchResult`
* **Function composition**: Accept iterable of scales instead of range tuple

## src/vidkompy/align/core.py

### Refactoring Opportunities

* **Decompose `_find_thumbnail_in_frames()`**: Split into pure helpers: `_prepare()`, `_analyse()`, `_aggregate()`
* **Progress handling cleanup**: Remove `progress` argument threading; let caller handle UI
* **Input validation**: Move `validate_inputs()` to shared `vidkompy.validation` module

## src/vidkompy/align/frame_extractor.py

### Strategy Pattern Implementation

* **Polymorphic extraction**: Merge `_extract_frames_from_video` and `_load_image_as_frames` behind unified `_extract()` strategy
* **Constants consolidation**: Move `VIDEO_EXTS` & `IMAGE_EXTS` to `constants.py` for reuse
* **Resize unification**: Replace manual resize with `utils.image.resize_frame` for consistency

## src/vidkompy/align/result_types.py

### Data Model Improvements

* **MatchResult cleanup**: Remove unused `bg_frame_idx`; add optional `metadata: dict` for extensibility
* **Required timing**: Make `processing_time` non-optional to surface missing instrumentation

## src/vidkompy/comp/alignment_engine.py

### Single Responsibility Improvements

* **Break monolithic `process()`**: Split into `analyse()`, `align()`, `compose()` methods
* **Flag consolidation**: Convert bool flags (`skip_spatial`, `trim`, `blend`) into `ProcessingOptions` dataclass
* **Overlay delegation**: Move ROI/blending logic to `VideoProcessor.overlay_frames()` to eliminate duplication

## src/vidkompy/comp/dtw_aligner.py

### Performance and Clarity

* **Numba optimization**: Extract fallback decision into `_try_numba()` helper
* **Progress standardization**: Use shared `utils.progress.create_bar()` instead of local Rich setup
* **API simplification**: Accept pre-computed `Fingerprints` iterable, return `list[FrameAlignment]` directly

## src/vidkompy/comp/temporal_alignment.py

### Method Consolidation

* **Mask parameter**: Collapse `align_frames` variants by accepting optional `mask` argument
* **Shared utilities**: Promote border-mask creation to `utils.masks.create_border_mask`

---

## 🌟 Cross-Cutting Improvements

### Planned Enhancements

* **Progress bars**: Standardize Rich progress construction in `utils.progress` helper
* **Constants module**: Create `vidkompy.constants` for file-type sets, default windows, epsilons
* **Documentation**: Adopt Google-style docstrings consistently; enforce via `pydocstyle` pre-commit
* **Testing**: Refactor unit tests to use parametrized fixtures for shared algorithm behavior

This plan focuses on the most impactful remaining improvements while maintaining incremental, focused changes.