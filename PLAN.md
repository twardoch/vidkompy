# Spatial Alignment Integration Plan

## Overview

Replace the simple spatial alignment implementation in `src/vidkompy/comp/` with the advanced, multi-algorithm spatial alignment from `src/vidkompy/align/`. This will provide better accuracy, robustness, and multi-scale detection capabilities while maintaining backward compatibility.

## Current State Analysis

### `comp` Module Spatial Alignment (TO BE REPLACED)
- **Location**: `src/vidkompy/comp/spatial_alignment.py`
- **Algorithm**: Single template matching with `cv2.matchTemplate`
- **Scaling**: Basic downscaling when fg > bg
- **Performance**: Fast but limited (~few milliseconds)
- **Return Type**: `SpatialAlignment(x_offset, y_offset, scale_factor, confidence)`

### `align` Module Spatial Alignment (TO BE INTEGRATED)
- **Location**: `src/vidkompy/align/` (ThumbnailFinder class)
- **Algorithms**: 6 different algorithms with precision levels 0-4
- **Scaling**: Multi-scale search with unity scale preference
- **Performance**: Configurable speed/accuracy tradeoff (1ms-200ms)
- **Return Type**: `ThumbnailResult` with rich analysis data

## Integration Plan

### Phase 1: Core Replacement

#### Step 1.1: Update Dependencies
- **File**: `src/vidkompy/comp/alignment_engine.py`
- **Action**: Replace `SpatialAligner` import with `ThumbnailFinder`
- **Code Change**:
```python
# Replace:
from .spatial_alignment import SpatialAligner

# With:
from vidkompy.align import ThumbnailFinder
from vidkompy.align.result_types import ThumbnailResult
```

#### Step 1.2: Replace SpatialAligner Instantiation
- **File**: `src/vidkompy/comp/alignment_engine.py`
- **Location**: Line 78 in `__init__` method
- **Action**: Replace `SpatialAligner()` with `ThumbnailFinder()`
- **Code Change**:
```python
# Replace:
self.spatial_aligner = SpatialAligner()

# With:
self.thumbnail_finder = ThumbnailFinder()
```

#### Step 1.3: Create Result Converter Function
- **File**: `src/vidkompy/comp/alignment_engine.py`
- **Action**: Add method to convert `ThumbnailResult` to `SpatialAlignment`
- **Code Addition**:
```python
def _convert_thumbnail_result(self, result: ThumbnailResult) -> SpatialAlignment:
    """Convert ThumbnailResult from align module to comp module's SpatialAlignment format.
    
    Args:
        result: ThumbnailResult from the align module
        
    Returns:
        SpatialAlignment compatible with existing comp module code
    """
    return SpatialAlignment(
        x_offset=result.x_thumb_in_bg,
        y_offset=result.y_thumb_in_bg,
        scale_factor=result.scale_fg_to_thumb / 100.0,  # Convert percentage to factor
        confidence=result.confidence
    )
```

#### Step 1.4: Update Spatial Alignment Call
- **File**: `src/vidkompy/comp/alignment_engine.py`
- **Location**: Line 236 in `_compute_spatial_alignment` method
- **Action**: Replace `spatial_aligner.align()` call with `thumbnail_finder.find_thumbnail()`
- **Code Change**:
```python
# Replace:
spatial_alignment = self.spatial_aligner.align(
    fg_sample_frame, bg_sample_frame
)

# With:
# Extract frame timestamps for proper temporal sampling
fg_timestamp = (fg_info.duration / 2) if fg_info.duration else 0
bg_timestamp = (bg_info.duration / 2) if bg_info.duration else 0

# Use the align module for spatial alignment with unity scale preference
thumbnail_result = self.thumbnail_finder.find_thumbnail(
    fg=fg_info.path,
    bg=bg_info.path,
    precision=2,  # Balanced mode (default ~25ms)
    unity_scale=True,  # Prefer unity scale for video composition
    num_frames=1,  # Single frame analysis for speed
    fg_timestamp=fg_timestamp,
    bg_timestamp=bg_timestamp,
    verbose=self.verbose
)

# Convert to expected format
spatial_alignment = self._convert_thumbnail_result(thumbnail_result)
```

#### Step 1.5: Add Configuration Parameters
- **File**: `src/vidkompy/comp/alignment_engine.py`
- **Action**: Add spatial alignment configuration to `__init__`
- **Code Addition**:
```python
def __init__(self, engine_type: str = "full", verbose: bool = False, 
             spatial_precision: int = 2, unity_scale: bool = True):
    """Initialize alignment engine.
    
    Args:
        engine_type: Type of temporal alignment engine ("full" or "mask")
        verbose: Enable verbose logging
        spatial_precision: Spatial alignment precision level (0-4)
        unity_scale: Prefer unity scale for spatial alignment
    """
    self.engine_type = engine_type
    self.verbose = verbose
    self.spatial_precision = spatial_precision
    self.unity_scale = unity_scale
    # ... rest of existing init code
```

#### Step 1.6: Update CLI Integration
- **File**: `src/vidkompy/comp/cli.py`
- **Action**: Add new CLI parameters for spatial alignment configuration
- **Code Addition**:
```python
@click.option(
    "--spatial-precision",
    type=click.IntRange(0, 4),
    default=2,
    help="Spatial alignment precision level (0=ballpark ~1ms, 2=balanced ~25ms, 4=precise ~200ms)"
)
@click.option(
    "--unity-scale/--no-unity-scale",
    default=True,
    help="Prefer unity scale (100%) for spatial alignment"
)
def comp(bg: str, fg: str, output: str = None, engine: str = "full", 
         spatial_precision: int = 2, unity_scale: bool = True, 
         verbose: bool = False, **kwargs):
    """Video composition with intelligent alignment."""
    # ... existing code ...
    
    # Pass new parameters to alignment engine
    alignment_engine = AlignmentEngine(
        engine_type=engine,
        verbose=verbose,
        spatial_precision=spatial_precision,
        unity_scale=unity_scale
    )
```

### Phase 2: Deprecation and Cleanup

#### Step 2.1: Deprecate Old Spatial Alignment Module
- **File**: `src/vidkompy/comp/spatial_alignment.py`
- **Action**: Add deprecation warning and maintain for backward compatibility
- **Code Addition**:
```python
import warnings

warnings.warn(
    "spatial_alignment.py is deprecated. Use vidkompy.align module instead.",
    DeprecationWarning,
    stacklevel=2
)
```

#### Step 2.2: Update Tests
- **Action**: Update all tests that reference the old spatial alignment
- **Files**: All test files in `tests/` that test spatial alignment
- **Changes**: Update test imports and expected behavior to match new implementation

#### Step 2.3: Update Documentation
- **Files**: README.md, docstrings, inline comments
- **Action**: Update references to reflect the new unified spatial alignment approach

### Phase 3: Enhanced Features (Optional)

#### Step 3.1: Multi-Frame Spatial Alignment
- **Enhancement**: Use multiple frames for more robust spatial alignment
- **Implementation**: Add `num_frames` parameter to alignment configuration
- **Benefits**: Better accuracy for videos with motion or compression artifacts

#### Step 3.2: Adaptive Precision
- **Enhancement**: Start with low precision, increase if confidence is low
- **Implementation**: Add automatic precision escalation logic
- **Benefits**: Optimal speed/accuracy balance

#### Step 3.3: Spatial Alignment Caching
- **Enhancement**: Cache spatial alignment results for repeated processing
- **Implementation**: Add file hash-based caching system
- **Benefits**: Faster re-processing of same video pairs

## File Impact Summary

### Files to Modify
1. **`src/vidkompy/comp/alignment_engine.py`** - Primary integration point
2. **`src/vidkompy/comp/cli.py`** - CLI parameter additions
3. **`src/vidkompy/comp/spatial_alignment.py`** - Deprecation warning (keep for compatibility)
4. **`tests/`** - Update tests to use new implementation

### Files to Keep Unchanged
1. **`src/vidkompy/align/`** - No changes needed, used as-is
2. **`src/vidkompy/comp/temporal_alignment.py`** - Interface remains compatible
3. **`src/vidkompy/comp/video_processor.py`** - No direct spatial alignment usage

## Benefits of Integration

### Immediate Benefits
- **Better Accuracy**: Multi-algorithm approach with automatic fallbacks
- **Multi-Scale Detection**: Finds thumbnails at various scales, not just 1:1
- **Robustness**: Multiple detection methods (template, feature, phase correlation)
- **Performance Options**: User-configurable speed/accuracy tradeoff

### Long-Term Benefits
- **Unified Codebase**: Single spatial alignment implementation across all modules
- **Maintainability**: Easier to improve and debug spatial alignment in one place
- **Extensibility**: Easy to add new algorithms to the align module
- **Quality**: Rich analysis data for debugging and quality metrics

## Implementation Timeline

1. **Day 1**: Phase 1 Steps 1.1-1.4 (Core replacement)
2. **Day 2**: Phase 1 Steps 1.5-1.6 (Configuration and CLI)
3. **Day 3**: Phase 2 (Testing, deprecation, documentation)
4. **Day 4**: Phase 3 (Optional enhancements)

## Risk Mitigation

### Backward Compatibility
- Keep old `spatial_alignment.py` with deprecation warning
- Maintain existing `SpatialAlignment` return type
- Preserve CLI interface while adding new options

### Testing Strategy
- Run existing test suite to ensure no regressions
- Add new tests for enhanced spatial alignment features
- Compare output quality before/after integration

### Rollback Plan
- Old spatial alignment code remains available
- Easy to revert by changing imports back
- Configuration flags allow gradual migration

## Success Criteria

- [ ] All existing tests pass
- [ ] Spatial alignment accuracy is equal or better than before
- [ ] CLI maintains backward compatibility
- [ ] Processing time is reasonable for default settings
- [ ] Integration is transparent to end users
- [ ] Code is cleaner and more maintainable