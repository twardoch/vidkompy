# TODO

<rationale>
Add `--match_time border` which is a new temporal alignment mode. Make it default. If `--match_time border` is provided, we perform border matching: 

When we have aligned the bg and fg spatially, we identify a border frame of thickness `--border` (defaulting to 8). The border has up to 4 sides: top, bottom, left, right. 

If the fg image touches an edge of the canvas, then we don't consider that edge a border (or, the border thickness is 0). 

If the bg image is partially visible from underneath a fg, we consider that edge a border (the border thickness is `--border`). 

We construct the mask shape, which is two rectangles, one inside of the other. 

- The outer rectangle is the same as the fg image box.
- The inner rectangle is smaller by the "border thickness" on each edge (either 0 or `--border`, for each edge separately). 

The mask shape is the outer rectangle minus the inner rectangle. 

1. We perform similarity matching only within the mask shape. 

2. The new option `--window` defaulting to 0, lets us specify a frame count which is used as +/– number of frames where, for each fg frame, we look at the numbered bg frames (full frames or only within the mask) and find the most similar one. So if I gave window 1, then for fg frame 0 we'd look at bg frames 0, 1. For fg frame 1 we'd look at bg frames 0, 1, 2. For fg frame 2 we'd look at bg frames 1, 2, 3. And so on. This is an extension of the existing frame matching.  

3. If `--blend` is provided, we construct a blend mask which would be some sort of clever gradient that gives a smooth transition between the inner mask rectangle and the outer mask rectangle. And for each frame we perform some kind of smooth blending of the fg onto the bg 

</rationale>




<spec>
Here is a detailed SPEC for the requested improvements.

### **SPECIFICATION: Advanced Temporal Alignment and Composition**

This document outlines the plan to introduce three new features: `--match_time border`, a sliding `--window` for frame matching, and a `--blend` option for smoother composition.

---

### **1. New Temporal Alignment Mode: `--match_time border`**

This will be the new default temporal alignment mode. It focuses the similarity search on the border regions of the foreground video, which is ideal for aligning videos where one is a picture-in-picture overlay.

#### **1.1. Rationale**

In many overlay scenarios, the central content of the foreground and background may differ significantly, while the "frame" or border area created by the overlay remains consistent. By concentrating the matching algorithm on this border, we can achieve more accurate temporal synchronization, ignoring potentially noisy or dissimilar main content.

#### **1.2. Implementation Plan**

##### **`src/vidkompy/models.py`**

1.  **Update `MatchTimeMode` Enum**: Add a new member `BORDER` and set it as the default. The existing `PRECISE` mode will be kept for frame-based matching.
2.  **Update `ProcessingOptions`**: Add a new field `border_thickness: int = 8`.

```python
// src/vidkompy/models.py

class MatchTimeMode(Enum):
    """Temporal alignment modes."""
    BORDER = "border" # New Default
    FAST = "fast"
    PRECISE = "precise"

@dataclass
class ProcessingOptions:
    """Options for video processing."""
    # ... existing options
    border_thickness: int = 8
    blend: bool = False
    window: int = 0
```

##### **`src/vidkompy/vidkompy.py`**

1.  **Update `main` function**:
    * Change the default value of `match_time` to `"border"`.
    * Add a new argument `--border` with a default of `8`.
    * Pass the `border` value down to the `AlignmentEngine`.

```python
// src/vidkompy/vidkompy.py

def main(
    # ...
    match_time: str = "border", # Changed default
    # ...
    border: int = 8, # New argument
    blend: bool = False, # New argument
    window: int = 0, # New argument
    # ...
):
    # ...
    engine.process(
        # ...
        time_mode=time_mode,
        border_thickness=border,
        blend=blend,
        window=window
        # ...
    )
```

##### **`src/vidkompy/core/temporal_alignment.py`**

1.  **Create Border Mask Function**: A new method, `_create_border_mask`, will be added.
    * **Input**: `SpatialAlignment` result, `fg_info`, `bg_info`, and `border_thickness`.
    * **Logic**:
        * It determines which sides of the foreground frame are not flush with the background frame's edges[cite: 148].
        * It constructs two rectangles: an outer one matching the foreground dimensions and an inner one inset by `border_thickness` only on the non-flush edges.
        * The mask will be the area of the outer rectangle minus the inner rectangle.
    * **Output**: A binary mask `np.ndarray`.
2.  **Modify Similarity Calculation**: The `_compute_frame_similarity` (or a new, dedicated function) will be modified to accept an optional `mask` argument. When a mask is provided, the similarity metric (both hash-based and SSIM) will be calculated *only* within the masked region.

##### **`src/vidkompy/core/alignment_engine.py`**

1.  **Update `process` method**: Accept the new `border_thickness`, `blend`, and `window` parameters.
2.  **Integrate Mask Generation**:
    * After `_compute_spatial_alignment` is called, the `AlignmentEngine` will call the new `_create_border_mask` method from the `TemporalAligner`.
    * This mask will then be passed to `_compute_temporal_alignment` when `time_mode` is `BORDER`.

---

### **2. Sliding Window Frame Matching: `--window`**

This feature restricts the frame search space, improving both speed and relevance by preventing the algorithm from matching temporally distant frames.

#### **2.1. Rationale**

For videos that are already roughly synchronized, searching the entire background video for a match for every foreground frame is computationally expensive and unnecessary. A sliding window (`--window`) dramatically narrows the search, assuming the correct match is near the corresponding frame index. This is a powerful optimization for the existing frame-matching logic[cite: 153].

#### **2.2. Implementation Plan**

##### **`src/vidkompy/core/dtw_aligner.py`**

1.  **Modify `_build_dtw_matrix`**: The core logic of the DTW aligner already uses a `window_constraint` which serves a similar purpose[cite: 45]. We will adapt this. The existing `Sakoe-Chiba band` is centered on the diagonal[cite: 57]. The new `--window` parameter will define this band width.
    * The loop `for j in range(j_start, j_end):` will be modified. `j_start` will be `max(1, i - window)` and `j_end` will be `min(n_bg + 1, i + window)`. This ensures that for foreground frame `i`, we only search background frames in the range `[i - window, i + window]`.

##### **`src/vidkompy/vidkompy.py`**

1.  **Add `--window` argument**: A new CLI argument `--window` will be added, defaulting to `0` (which will imply the existing default, e.g. 100 in DTW, should be used). A non-zero value will override it.

##### **`src/vidkompy/core/alignment_engine.py`**

1.  **Pass `window` to `TemporalAligner`**: The `window` parameter will be passed through the `process` method to the `_compute_temporal_alignment` method.

---

### **3. Frame Blending: `--blend`**

This feature will enable smooth, gradient-based blending at the edges of the overlay, creating a more professional and seamless composite.

#### **3.1. Rationale**

A hard-edged overlay can be visually jarring. Blending the foreground frame's borders with the background creates a subtle, feathered transition, making the composite look more natural and integrated.

#### **3.2. Implementation Plan**

##### **`src/vidkompy/vidkompy.py`**

1.  **Add `--blend` flag**: A boolean flag `--blend` will be added to the `main` function, defaulting to `False`.

##### **`src/vidkompy/core/alignment_engine.py`**

1.  **Create `_create_blend_mask` method**:
    * **Input**: `SpatialAlignment` result, border information (similar to the border matching mask).
    * **Logic**: This method will generate a multi-channel alpha mask with the same dimensions as the foreground frame.
        * The interior of the frame (inside the border region) will be fully opaque (alpha=1.0).
        * The exterior (outside the frame) will be fully transparent (alpha=0.0).
        * The border region will contain a smooth gradient (e.g., linear or Gaussian) from opaque to transparent. This will be calculated for each edge that is not flush with the background.
    * **Output**: An alpha mask `np.ndarray` with values from `0.0` to `1.0`.

2.  **Modify `_overlay_frames` method**:
    * This method will be updated to accept an optional `blend_mask`.
    * If a `blend_mask` is provided, instead of a direct pixel replacement, it will perform alpha blending:
        `composite = (fg_frame * blend_mask) + (bg_slice * (1 - blend_mask))`
    * The `bg_slice` refers to the region of the background frame directly underneath the foreground frame.

3.  **Update `_compose_with_opencv`**: If blending is enabled, it will first generate the blend mask and then pass it to `_overlay_frames` for each frame in the composition loop.
</spec>

