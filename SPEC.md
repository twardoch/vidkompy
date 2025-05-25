# SPEC: Fix Temporal Drift in Video Alignment

## Problem Analysis

Based on the git history and current code state, the drift issue was introduced when the precise alignment engine was added. The issue manifests around 5 seconds in the test video where the background video speeds up while the foreground maintains normal speed, causing a ~1 second delay in hand movements.

### Root Causes Identified

1. **Drift Correction Interval**: The `drift_correction_interval` parameter in `PreciseEngineConfig` is set to 32 frames by default, which may be too small and causing overcorrection.

2. **Missing Spatial Cropping**: The temporal alignment is comparing full-resolution frames, but the foreground is typically smaller than the background. We should crop the background frames to match the foreground region for more accurate temporal alignment.

3. **Resolution Mismatch**: The frame extraction uses a `resize_factor=0.25` but doesn't account for the spatial alignment offset when comparing frames.

## Proposed Solution

### Step 1: Implement Spatial Cropping for Temporal Alignment

**Rationale**: When comparing frames for temporal alignment, we should only compare the overlapping regions. This will eliminate false dissimilarities caused by background content outside the foreground area.

**Implementation**:
1. First perform spatial alignment to get x/y offsets
2. During frame extraction for temporal alignment, crop background frames to the foreground region
3. Pass cropped frames to the temporal alignment engine

### Step 2: Adjust Drift Correction Parameters

**Rationale**: The current drift correction interval of 32 frames is too aggressive, causing overcorrection that manifests as the background "speeding up".

**Implementation**:
1. Increase `drift_correction_interval` from 32 to 100 frames
2. Reduce the blend factor in drift correction from 0.7 to 0.85 (trust original mapping more)
3. Add validation to ensure monotonic progression without aggressive jumps

### Step 3: Fix Frame Extraction Resolution

**Rationale**: The resize factor should be consistent between spatial and temporal alignment phases.

**Implementation**:
1. Use the same resize factor for both spatial and temporal alignment
2. Ensure the cropping is applied after resizing to maintain consistent coordinates

## Implementation Plan

### Phase 1: Add Spatial Cropping to Temporal Alignment

1. Modify `_align_frames_precise` in `temporal_alignment.py`:
   - Add spatial alignment step before frame extraction
   - Pass spatial offset to frame extraction
   - Implement frame cropping based on spatial alignment

2. Update `extract_all_frames` in `video_processor.py`:
   - Add optional crop parameters (x, y, width, height)
   - Apply cropping during frame extraction if parameters provided

### Phase 2: Tune Drift Correction

1. Modify `PreciseEngineConfig` defaults:
   - Change `drift_correction_interval` from 32 to 100
   - Add `drift_blend_factor` parameter (default 0.85)

2. Update `apply_drift_correction` in `multi_resolution_aligner.py`:
   - Use the new blend factor parameter
   - Add logging to track drift correction amounts
   - Implement smoother transitions between segments

### Phase 3: Validation and Testing

1. Add drift detection metrics:
   - Calculate frame-to-frame temporal differences
   - Log warnings when drift exceeds threshold
   - Output drift statistics in verbose mode

2. Test with the problematic video:
   - Verify hand movements stay synchronized
   - Check that background doesn't speed up at 5-second mark
   - Ensure overall alignment quality is maintained

## Code Changes Overview

### 1. temporal_alignment.py
```python
def _align_frames_precise(self, bg_info, fg_info, trim):
    # NEW: Perform spatial alignment first
    spatial_result = self.spatial_aligner.align(bg_info, fg_info)
    x_offset, y_offset = spatial_result.offset
    
    # Extract frames with cropping
    bg_frames = self.processor.extract_all_frames(
        bg_info.path, 
        resize_factor=0.25,
        crop=(x_offset, y_offset, fg_info.width, fg_info.height)
    )
    fg_frames = self.processor.extract_all_frames(
        fg_info.path, 
        resize_factor=0.25
    )
```

### 2. video_processor.py
```python
def extract_all_frames(self, video_path, resize_factor=1.0, crop=None):
    # Add cropping support
    if crop:
        x, y, w, h = crop
        # Apply crop filter in ffmpeg command
```

### 3. multi_resolution_aligner.py
```python
@dataclass
class PreciseEngineConfig:
    drift_correction_interval: int = 100  # Increased from 32
    drift_blend_factor: float = 0.85  # New parameter
```

## Expected Outcomes

1. **Eliminated Drift**: The background video should maintain consistent speed relative to the foreground
2. **Better Accuracy**: Cropping to overlapping regions should improve temporal matching accuracy
3. **Smoother Transitions**: Less aggressive drift correction should prevent sudden speed changes
4. **Maintained Performance**: Changes should not significantly impact processing time

## Risk Mitigation

1. **Fallback Mechanism**: Keep the current direct mapping mode as a fallback
2. **Configurable Parameters**: Make drift correction tunable via CLI flags
3. **Extensive Logging**: Add detailed logs to track alignment decisions
4. **Gradual Rollout**: Test thoroughly before making it the default engine