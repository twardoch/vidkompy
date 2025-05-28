# Refactoring Plan

Below is a **first-pass refactoring map**.
H2 sections list files; H3 subsections drill down to concrete objects that deserve work.
Actions are phrased imperatively so they read like commit messages.&#x20;

---

## src/vidkompy/utils/**init**.py

### src/vidkompy/utils/**init**.py\:module-level

* Drop logging constants export; re-export them from `vidkompy.logging` to keep utilities I/O-free
* Generate `__all__` dynamically via `__getattr__`, avoiding manual drift
* Move rich-console imports to call sites to cut import time

## src/vidkompy/utils/logging.py

### src/vidkompy/utils/logging.py\:LOG\_FORMAT\_VERBOSE / LOG\_FORMAT\_DEFAULT

* Rename to `FORMAT_VERBOSE`, `FORMAT_COMPACT`—the `LOG_` prefix is redundant inside a logging module
* Add `make_logger(name, verbose=False)` helper to centralise logger wiring

## src/vidkompy/utils/correlation.py

### src/vidkompy/utils/correlation.py\:compute\_normalized\_correlation

* Replace hand-rolled mean/var check with `np.nan_to_num` + `np.corrcoef` for clarity
* Hoist NaN/variance guard into a private `_safe_corr()` used by both public functions

### src/vidkompy/utils/correlation.py\:histogram\_correlation

* De-duplicate normalisation code by calling shared helper above
* Expose a `CORR_EPS = 1e-7` constant; avoid scattering magic numbers

## src/vidkompy/align/algorithms.py

### src/vidkompy/align/algorithms.py\:TemplateMatchingAlgorithm.\_match\_at\_scale

* Delegate scale-resize & bounds check to `utils.image.resize_and_validate()`
* Return early on invalid scale instead of constructing zero-confidence `MatchResult`

### src/vidkompy/align/algorithms.py\:TemplateMatchingAlgorithm.match\_at\_multiple\_scales

* Extract scale-grid generation to `_iter_scales()`; removes duplication with parallel method
* Accept iterable of scales instead of range tuple to make function composition-friendly

### src/vidkompy/align/algorithms.py\:FeatureMatchingAlgorithm.enhanced\_feature\_matching

* Split into `_choose_detector()`, `_match_descriptors()`, `_estimate_transform()` helpers
* Introduce enum-to-cv2-factory map instead of `if/elif` chain
* Separate detector selection («strategy») from execution («policy»)

### src/vidkompy/align/algorithms.py\:SubPixelRefinementAlgorithm.\_calculate\_subpixel\_correlation

* Replace ad-hoc normalisation with call to shared `_safe_corr()`
* Remove silent bare `except`; surface unexpected errors in verbose mode

## src/vidkompy/align/precision.py

### src/vidkompy/align/precision.py\:SCALE\_PARAMS

* Replace dict-of-lambdas with `namedtuple ScaleParams(range_fn, steps)` for type safety
* Store constant window sizes instead of recomputing nested lambdas each call

### src/vidkompy/align/precision.py\:PrecisionAnalyzer.\_get\_algorithm

* Convert to `@functools.cached_property` per algorithm; simplifies caching logic
* Raise `NotImplementedError` instead of generic `ValueError` for unknown key

## src/vidkompy/align/core.py

### src/vidkompy/align/core.py\:ThumbnailFinder.\_find\_thumbnail\_in\_frames

* Decompose into three pure helpers: `_prepare()`, `_analyse()`, `_aggregate()`
* Remove `progress` argument threading; let caller handle UI

### src/vidkompy/align/core.py\:ThumbnailFinder.validate\_inputs

* Move to `vidkompy.validation` module so comp layer can reuse

## src/vidkompy/align/frame\_extractor.py

### src/vidkompy/align/frame\_extractor.py\:FrameExtractor.extract\_frames

* Merge `_extract_frames_from_video` and `_load_image_as_frames` behind polymorphic `_extract()` strategy
* Promote `VIDEO_EXTS` & `IMAGE_EXTS` to `constants.py`; reuse in `comp.video_processor`

### src/vidkompy/align/frame\_extractor.py\:FrameExtractor.preprocess\_frame

* Replace manual resize path with `utils.image.resize_frame`; ensure single resize implementation

## src/vidkompy/align/result\_types.py

### src/vidkompy/align/result\_types.py\:MatchResult

* Remove `bg_frame_idx` (unused outside DTW path); add optional `metadata: dict` for extensibility
* Make `processing_time` non-optional; defaulting to `0.0` hides missing instrumentation

## src/vidkompy/comp/domain\_models.py

### src/vidkompy/comp/domain\_models.py\:VideoInfo

* Extract nested `AudioInfo` dataclass (sample rate, channels) to reduce nullable fields
* Provide `@classmethod from_path()` to encapsulate ffprobe call currently in `VideoProcessor`

### src/vidkompy/comp/domain\_models.py\:SpatialAlignment

* Rename to `SpatialTransform`; clarify that scaling + translation forms a 2-D similarity transform
* Add `as_matrix()` helper returning 3 × 3 homography for downstream compositors

## src/vidkompy/comp/enums.py

### src/vidkompy/comp/enums.py\:TemporalMethod

* Remove alias `FRAMES`; prefer canonical `CLASSIC`; update call sites
* Document enum members with brief docstrings instead of inline comments

## src/vidkompy/comp/alignment\_engine.py

### src/vidkompy/comp/alignment\_engine.py\:AlignmentEngine.process

* Break monolithic `process()` into `analyse()`, `align()`, `compose()` for SRP
* Convert many bool flags (`skip_spatial`, `trim`, `blend`) into small `ProcessingOptions` dataclass

### src/vidkompy/comp/alignment\_engine.py\:AlignmentEngine.\_overlay\_frames

* Delegate overlay logic (ROI, blending) to `VideoProcessor.overlay_frames()`; avoids duplicate math with `_compose_with_opencv`

## src/vidkompy/comp/dtw\_aligner.py

### src/vidkompy/comp/dtw\_aligner.py\:DTWAligner.\_build\_dtw\_matrix

* Extract Numba-fallback decision into `_try_numba()`; short-circuit main loop
* Use shared `utils.progress.create_bar()` instead of local Rich setup

### src/vidkompy/comp/dtw\_aligner.py\:DTWAligner.align\_videos

* Accept iterable of pre-computed `Fingerprints` to decouple I/O from algorithm
* Return `list[FrameAlignment]` directly; drop tuple intermediary step

## src/vidkompy/comp/temporal\_alignment.py  *(not shown in snippet)*

### src/vidkompy/comp/temporal\_alignment.py\:TemporalAligner.align\_frames

* Collapse variant methods (`align_frames`, `align_frames_with_mask`) by accepting optional `mask` arg
* Promote shared border-mask creation to `utils.masks.create_border_mask`

---

### Cross-cutting clean-ups

* **Terminology** – Replace mixed use of “thumbnail”, “spatial alignment”, “template” with “SpatialTransform” consistently across `align` and `comp` layers
* **Percent vs factor** – Store scales internally as *factor* (1.0 == unity); convert to percent only in UI
* **Verbose flag** – Centralise into `Config(verbose: bool)` singleton; remove per-class duplication
* **Progress bars** – Move Rich progress construction to `utils.progress` helper; standardise look & feel
* **Constants & paths** – Add `vidkompy.constants` for file-type sets, default windows, epsilons, etc.
* **Docstrings** – Adopt Google-style docstrings across modules; enforce via `pydocstyle` pre-commit
* **Tests** – Refactor unit tests to use parametrised fixtures for algorithms that share behaviour

This plan keeps changes incremental and focused, paving the way for a unified, leaner codebase.
