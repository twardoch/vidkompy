# SPEC4: Vidkompy Code Refactoring and Performance Optimization Plan

## Executive Summary

This specification outlines a comprehensive refactoring plan to transform vidkompy from a complex, multi-path codebase into a lean, high-performance video alignment tool. The refactoring will remove ~60% of the code while improving performance by 10-50x through strategic use of modern libraries.

## Core Principles

1. **One Path, One Fallback**: Each operation has a primary implementation and exactly one fallback
2. **Performance First**: Replace slow algorithms with GPU-accelerated alternatives
3. **Remove Redundancy**: Eliminate duplicate functionality and alternative implementations
4. **Simplify Configuration**: Reduce CLI options to essential parameters only

## Part 1: Remove Alternative Implementations

### 1.1 Temporal Alignment - Keep Only DTW with FAISS

**REMOVE:**
```python
# From temporal_alignment.py - DELETE ENTIRE METHODS:
- _find_keyframe_matches()  # Lines 373-623 - Classic keyframe matching
- _build_cost_matrix()      # Lines 625-783 - Slow SSIM-based matching
- _find_optimal_path()      # Lines 785-865 - Redundant with DTW
- _refine_matches()         # Lines 867-925 - Band-aid for poor matching
- _filter_monotonic()       # Lines 927-953 - Not needed with DTW
- _interpolate_bg_frame()   # Lines 1007-1079 - Linear interpolation
- align_audio()             # Lines 195-230 - Audio alignment
- _create_audio_based_alignment() # In alignment_engine.py
- _create_direct_mapping()  # Lines 1141-1165 - Fallback we don't need
```

**REPLACE WITH:**
```python
class TemporalAligner:
    def __init__(self, processor: VideoProcessor):
        self.processor = processor
        self.faiss_index = None  # FAISS index for fast similarity search
        
    def align_frames(self, bg_info: VideoInfo, fg_info: VideoInfo) -> TemporalAlignment:
        """Single method using FAISS + DTW"""
        # Extract frame embeddings using pre-trained CNN
        fg_embeddings = self._extract_embeddings(fg_info)
        bg_embeddings = self._extract_embeddings(bg_info)
        
        # Build FAISS index for background frames
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        self.faiss_index.add(bg_embeddings)
        
        # Find nearest neighbors for each foreground frame
        k = 5  # Search top-5 candidates
        distances, indices = self.faiss_index.search(fg_embeddings, k)
        
        # Apply DTW on top candidates only
        return self._refine_with_dtw(indices, distances)
```

### 1.2 Spatial Alignment - Keep Only Phase Correlation

**REMOVE:**
```python
# From spatial_alignment.py - DELETE:
- _template_matching()  # Lines 89-125 - Slower than phase correlation
- _feature_matching()   # Lines 127-225 - ORB matching is unreliable
```

**REPLACE WITH:**
```python
class SpatialAligner:
    def align(self, bg_frame: np.ndarray, fg_frame: np.ndarray) -> SpatialAlignment:
        """Single method using phase correlation with cuFFT"""
        # Resize if needed
        if self._needs_scaling(bg_frame, fg_frame):
            fg_frame, scale = self._scale_to_fit(bg_frame, fg_frame)
        else:
            scale = 1.0
            
        # Phase correlation using GPU
        shift = self._phase_correlate_gpu(bg_frame, fg_frame)
        
        # Sub-pixel refinement
        refined_shift = self._refine_subpixel(bg_frame, fg_frame, shift)
        
        return SpatialAlignment(
            x_offset=int(refined_shift[0]),
            y_offset=int(refined_shift[1]),
            scale_factor=scale,
            confidence=0.95
        )
```

### 1.3 Frame Fingerprinting - Use Single Fast Method

**REMOVE:**
```python
# From frame_fingerprint.py - DELETE:
- All multiple hash algorithms (phash, ahash, dhash, mhash)
- compute_fingerprint() with dictionary of hashes
- compare_fingerprints() with weighted combination
- _compute_color_histogram()
```

**REPLACE WITH:**
```python
class FrameEmbedder:
    def __init__(self):
        # Use MobileNetV3 for fast feature extraction
        self.model = torch.jit.load('mobilenet_v3_small.pt')
        self.model.eval().cuda()
        
    def extract_embeddings(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Extract 256-dim embeddings using MobileNet"""
        # Batch process on GPU
        batch = torch.stack([self._preprocess(f) for f in frames])
        with torch.no_grad():
            embeddings = self.model(batch.cuda())
        return embeddings.cpu().numpy()
```

## Part 2: Simplify Architecture

### 2.1 Merge Overlapping Classes

**REMOVE:**
- `DTWAligner` class - merge into `TemporalAligner`
- `FrameFingerprinter` class - replace with `FrameEmbedder`
- `ProcessingOptions` dataclass - use simple dict

**NEW STRUCTURE:**
```
vidkompy/
├── core/
│   ├── __init__.py
│   ├── video_io.py          # Merged video_processor.py functionality
│   ├── spatial_aligner.py   # Phase correlation only
│   ├── temporal_aligner.py  # FAISS + DTW only
│   └── compositor.py        # Frame composition logic
├── models.py                # Simplified data classes
└── vidkompy.py             # Simplified CLI
```

### 2.2 Remove Unnecessary Modes and Options

**REMOVE FROM CLI:**
```python
# Current complex CLI arguments to remove:
- match_time (keep only precise mode)
- temporal_align (no choice, always use optimized DTW)
- match_space (no choice, always use phase correlation)
- skip_spatial_align (always align)
- max_keyframes (auto-determine based on video length)
- window (use adaptive window)
```

**NEW SIMPLIFIED CLI:**
```python
def main(
    bg: str,
    fg: str,
    output: str | None = None,
    border: int = 8,      # Keep for border mode
    blend: bool = False,  # Keep for smooth edges
    gpu: bool = True,     # GPU acceleration on/off
    verbose: bool = False
):
    """Simplified interface with only essential options"""
```

## Part 3: Optimize Border Mode

### 3.1 Smart Border Mask Optimization

**CURRENT ISSUE:** Border mask is computed for entire frame even when some edges are zero-width.

**OPTIMIZE:**
```python
def create_optimized_border_mask(spatial_alignment, fg_info, bg_info, thickness=8):
    """Create minimal border mask based on actual visible borders"""
    
    # Detect which borders are visible
    visible_borders = {
        'top': spatial_alignment.y_offset > 0,
        'bottom': (spatial_alignment.y_offset + fg_info.height) < bg_info.height,
        'left': spatial_alignment.x_offset > 0,
        'right': (spatial_alignment.x_offset + fg_info.width) < bg_info.width
    }
    
    # If 3 borders have 0 width, optimize to single rectangle
    visible_count = sum(visible_borders.values())
    if visible_count == 1:
        # Return minimal rectangle for the single visible border
        return self._create_single_border_rect(visible_borders, thickness)
    
    # Otherwise create standard mask
    return self._create_full_border_mask(visible_borders, thickness)
```

## Part 4: Performance Optimizations

### 4.1 Replace OpenCV with Faster Alternatives

**REMOVE:**
- `cv2.VideoCapture` / `cv2.VideoWriter`
- `cv2.resize` for frame scaling
- `cv2.matchTemplate` (already removing)

**REPLACE WITH:**
```python
# Use PyAV for faster video I/O
import av

class VideoIO:
    def read_frames_batch(self, path: str, indices: List[int], batch_size=32):
        """Batch read frames using PyAV with hardware decoding"""
        container = av.open(path)
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = 'NONKEY'  # Fast seek
        
        # Enable hardware decoding
        if self.gpu:
            stream.codec_context.options = {'hwaccel': 'nvdec'}
```

### 4.2 Implement FAISS for Frame Matching

**NEW IMPLEMENTATION:**
```python
import faiss
import torch
from torchvision import models, transforms

class FAISSMatcher:
    def __init__(self, embedding_dim=256):
        self.dimension = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Use GPU if available
        if faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, self.index
            )
    
    def build_index(self, embeddings: np.ndarray):
        """Build FAISS index from embeddings"""
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
    
    def search(self, queries: np.ndarray, k: int = 5):
        """Search k nearest neighbors"""
        faiss.normalize_L2(queries)
        distances, indices = self.index.search(queries, k)
        return distances, indices
```

### 4.3 GPU-Accelerated Phase Correlation

**IMPLEMENT:**
```python
import cupy as cp
from cupyx.scipy.fft import fft2, ifft2

class GPUPhaseCorrelator:
    def correlate(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[float, float]:
        """Phase correlation using CuPy for GPU acceleration"""
        # Transfer to GPU
        gpu_img1 = cp.asarray(img1, dtype=cp.float32)
        gpu_img2 = cp.asarray(img2, dtype=cp.float32)
        
        # FFT on GPU
        f1 = fft2(gpu_img1)
        f2 = fft2(gpu_img2)
        
        # Cross-power spectrum
        cross_power = (f1 * cp.conj(f2)) / cp.abs(f1 * cp.conj(f2))
        
        # Inverse FFT
        correlation = cp.real(ifft2(cross_power))
        
        # Find peak
        peak = cp.unravel_index(cp.argmax(correlation), correlation.shape)
        
        # Sub-pixel refinement using parabolic fit
        subpixel_peak = self._refine_peak(correlation, peak)
        
        return subpixel_peak.get()  # Transfer back to CPU
```

## Part 5: Library Integration

### 5.1 Required New Dependencies

```toml
# pyproject.toml additions
dependencies = [
    # Core libraries (keep existing)
    "numpy>=1.24.0",
    "fire>=0.5.0",
    "rich>=13.0.0",
    "loguru>=0.7.0",
    
    # New high-performance libraries
    "torch>=2.0.0",          # For neural embeddings
    "torchvision>=0.15.0",   # Pre-trained models
    "faiss-gpu>=1.7.3",      # Fast similarity search
    "cupy-cuda12x>=12.0.0",  # GPU arrays and FFT
    "av>=10.0.0",            # PyAV for fast video I/O
    
    # Remove these
    # "opencv-contrib-python",  # Replace with specialized libs
    # "scipy",                  # Replace with CuPy
    # "scikit-image",          # Not needed
    # "soundfile",             # Remove audio support
]
```

### 5.2 Performance Targets

| Operation | Current | Target | Method |
|-----------|---------|--------|--------|
| Frame extraction | 50 fps | 500 fps | PyAV + batch processing |
| Frame similarity | 3 fps | 300 fps | FAISS on GPU |
| Spatial alignment | 100 fps | 1000 fps | CuPy phase correlation |
| Video composition | 30 fps | 300 fps | Batch processing + GPU |

## Part 6: Implementation Order

### Phase 1: Core Refactoring (Week 1)
1. **Day 1-2**: Remove all alternative implementations
   - Delete classic temporal alignment methods
   - Delete template/feature matching
   - Delete audio alignment code
   
2. **Day 3-4**: Simplify architecture
   - Merge classes
   - Simplify data models
   - Clean up CLI interface
   
3. **Day 5**: Create new project structure
   - Reorganize files
   - Update imports
   - Fix tests

### Phase 2: Performance Implementation (Week 2)
1. **Day 1-2**: Implement FAISS integration
   - Create FrameEmbedder with MobileNet
   - Implement FAISSMatcher
   - Update TemporalAligner
   
2. **Day 3-4**: GPU acceleration
   - Implement GPUPhaseCorrelator
   - Add PyAV video I/O
   - Batch processing pipeline
   
3. **Day 5**: Border optimization
   - Implement smart border detection
   - Optimize mask generation
   - Test performance

### Phase 3: Testing and Optimization (Week 3)
1. **Day 1-2**: Performance benchmarking
   - Create benchmark suite
   - Profile GPU utilization
   - Identify bottlenecks
   
2. **Day 3-4**: Edge case handling
   - Test with various video formats
   - Handle GPU memory limits
   - Add CPU fallbacks
   
3. **Day 5**: Documentation
   - Update README
   - Create migration guide
   - Performance tuning guide

## Code Removal Summary

**Files to delete entirely:**
- `dtw_aligner.py` (merge minimal parts into temporal_aligner.py)
- `frame_fingerprint.py` (replace with embeddings)

**Methods to remove (line counts are approximate):**
- `temporal_alignment.py`: Remove ~800 lines (keeping only ~200)
- `spatial_alignment.py`: Remove ~150 lines (keeping only ~50)
- `alignment_engine.py`: Remove ~200 lines of audio/fallback code
- `vidkompy.py`: Remove ~100 lines of CLI parsing

**Total code reduction: ~60% (from ~3000 to ~1200 lines)**

## Configuration Migration

### Old Configuration (REMOVE):
```python
# Too many options, confusing for users
main(bg, fg, output, match_time="border", match_space="precise", 
     temporal_align="dtw", skip_spatial_align=False, trim=True,
     verbose=False, max_keyframes=200, border=8, blend=False, window=0)
```

### New Configuration (KEEP):
```python
# Simple, focused on essentials
main(bg, fg, output=None, border=8, blend=False, gpu=True, verbose=False)
```

## Performance Benchmarks to Track

Create `benchmark.py` to track improvements:

```python
BENCHMARKS = {
    "8_second_video": {
        "before": {"time": 210.0, "fps": 2.3},  # 3.5 minutes
        "target": {"time": 10.0, "fps": 48.3},  # 10 seconds
    },
    "spatial_alignment_1080p": {
        "before": {"time": 0.010, "accuracy": 0.95},
        "target": {"time": 0.001, "accuracy": 0.99},
    },
    "temporal_alignment_500_frames": {
        "before": {"time": 180.0, "method": "ssim"},
        "target": {"time": 2.0, "method": "faiss"},
    }
}
```

## Conclusion

This refactoring plan transforms vidkompy from a feature-rich but slow tool into a focused, high-performance video alignment system. By removing redundant code paths and leveraging modern GPU-accelerated libraries, we achieve both simplicity and speed. The new architecture is easier to maintain, test, and extend while delivering 10-50x performance improvements across all operations.




## Major Changes:

1. **Remove 60% of the code** by eliminating alternative implementations:
   - Delete classic keyframe matching (~600 lines)
   - Remove audio alignment entirely
   - Keep only phase correlation for spatial alignment
   - Use FAISS + DTW as the single temporal alignment method

2. **Simplify the CLI** from 10+ parameters to just 4 essential ones:
   ```python
   # Before: Confusing with many options
   main(bg, fg, output, match_time="border", match_space="precise", 
        temporal_align="dtw", skip_spatial_align=False, trim=True, ...)
   
   # After: Clean and focused
   main(bg, fg, output=None, border=8, blend=False, gpu=True, verbose=False)
   ```

3. **Integrate high-performance libraries**:
   - **FAISS** for 100x faster frame matching
   - **CuPy** for GPU-accelerated phase correlation
   - **PyAV** for 10x faster video I/O
   - **PyTorch MobileNet** for efficient frame embeddings

4. **Optimize border mode** as requested - when 3 borders have 0-width, create a minimal rectangle instead of a full mask

5. **Clear implementation timeline**: 3-week plan with daily tasks that a junior developer can follow

The plan includes specific line numbers to delete, exact code replacements, and performance targets showing 10-50x improvements across all operations. This will transform vidkompy from a slow, complex tool into a fast, focused one that's easier to maintain.