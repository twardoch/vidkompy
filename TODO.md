# TODO

Fix this and then try to run `./benchmark.sh` to test your fix. 

```
2025-05-25 23:19:56.597 | WARNING  | vidkompy.core.multi_resolution_aligner:<module>:27 - Numba optimizations not available for multi-resolution alignment
23:19:56 | INFO     | main - CLI options used:
23:19:56 | INFO     | main -   Background video: tests/bg.mp4
23:19:56 | INFO     | main -   Foreground video: tests/fg.mp4
23:19:56 | INFO     | main -   Output path: tests/q-tunnel_mask-w30.mp4
23:19:56 | INFO     | main -   Engine: tunnel_mask
23:19:56 | INFO     | main -   Drift interval: 100
23:19:56 | INFO     | main -   Window: 30
23:19:56 | INFO     | main -   Margin: 8
23:19:56 | INFO     | main -   Smooth blending: False
23:19:56 | INFO     | main -   GPU acceleration: False
23:19:56 | INFO     | main -   Verbose logging: True
23:19:56 | INFO     | __init__ - ✓ Perceptual hashing enabled (pHash)
23:19:56 | INFO     | process - Analyzing videos...
23:19:56 | DEBUG    | get_video_info - Probing video: tests/bg.mp4
23:19:57 | INFO     | get_video_info - Video info for bg.mp4: 1920x1080, 60.00 fps, 7.85s, 472 frames, audio: yes
23:19:57 | DEBUG    | get_video_info - Probing video: tests/fg.mp4
23:19:57 | INFO     | get_video_info - Video info for fg.mp4: 1920x870, 60.89 fps, 8.04s, 483 frames, audio: yes
23:19:57 | INFO     | _log_compatibility - Video compatibility check:
23:19:57 | INFO     | _log_compatibility -   Resolution: 1920x1080 vs 1920x870
23:19:57 | INFO     | _log_compatibility -   FPS: 60.00 vs 60.89
23:19:57 | INFO     | _log_compatibility -   Duration: 7.85s vs 8.04s
23:19:57 | INFO     | _log_compatibility -   Audio: yes vs yes
23:19:57 | INFO     | process - Computing spatial alignment...
23:19:58 | DEBUG    | _template_matching - Using template matching for spatial alignment
23:19:58 | INFO     | _template_matching - Template match found at (0, 0) with confidence 0.941
23:19:58 | INFO     | process - Spatial alignment result: offset=(0, 0), scale=1.000, confidence=0.941
23:19:58 | INFO     | process - Computing temporal alignment...
23:19:58 | INFO     | align_frames - Starting frame-based temporal alignment
23:19:58 | INFO     | align_frames - Using tunnel_mask temporal alignment engine
23:19:58 | INFO     | _align_frames_tunnel - Performing spatial alignment for tunnel engine...
23:19:58 | DEBUG    | _template_matching - Using template matching for spatial alignment
23:19:58 | INFO     | _template_matching - Template match found at (0, 0) with confidence 0.941
23:19:58 | INFO     | _align_frames_tunnel - Spatial offset: (0, 0)
23:19:58 | INFO     | _align_frames_tunnel - Extracting all frames for tunnel alignment...
23:19:58 | INFO     | extract_all_frames - Extracting all 483 frames at 480x217
23:20:00 | INFO     | extract_all_frames - Extracted 483 frames from fg.mp4
23:20:00 | INFO     | extract_all_frames - Extracting all 472 frames at 480x270
23:20:03 | INFO     | extract_all_frames - Extracted 472 frames from bg.mp4
23:20:03 | INFO     | _align_frames_tunnel - Performing tunnel_mask alignment...
23:20:03 | INFO     | align - Generated mask with 95.4% coverage
23:20:03 | INFO     | align - Starting tunnel alignment with 483 FG and 472 BG frames
23:20:03 | INFO     | align - Config: window_size=30, downsample=0.5
23:20:04 | INFO     | align - Downsampled to 483 FG and 472 BG frames
23:20:04 | DEBUG    | _forward_pass - Forward: FG 0 -> BG 0 (diff=11.626)
23:20:10 | DEBUG    | _forward_pass - Forward: FG 100 -> BG 98 (diff=5.191)
23:20:19 | DEBUG    | _forward_pass - Forward: FG 200 -> BG 196 (diff=4.169)
23:20:27 | DEBUG    | _forward_pass - Forward: FG 300 -> BG 298 (diff=6.616)
23:20:36 | DEBUG    | _forward_pass - Forward: FG 400 -> BG 399 (diff=4.957)
23:20:46 | DEBUG    | _backward_pass - Backward: FG 400 -> BG 399 (diff=4.957)
23:20:52 | DEBUG    | _backward_pass - Backward: FG 300 -> BG 298 (diff=6.616)
23:20:57 | DEBUG    | _backward_pass - Backward: FG 200 -> BG 196 (diff=4.169)
23:21:02 | DEBUG    | _backward_pass - Backward: FG 100 -> BG 98 (diff=5.191)
23:21:06 | DEBUG    | _backward_pass - Backward: FG 0 -> BG 0 (diff=11.626)
23:21:06 | INFO     | _merge_mappings - Merge: avg difference = 0.00, confidence = 1.000
23:21:06 | INFO     | align - Cache stats: 0 hits, 0 misses
23:21:06 | INFO     | align - Final confidence: 1.000
23:21:06 | ERROR    | main - Processing failed: FrameAlignment.__init__() got an unexpected keyword argument 'fg_frame'
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/src/vidkompy/__main__.py", line 15, in <module>
    cli()
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/src/vidkompy/__main__.py", line 11, in cli
    fire.Fire(main)
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/fire/core.py", line 135, in Fire
    component_trace = _Fire(component, args, parsed_flag_args, context, name)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/fire/core.py", line 468, in _Fire
    component, remaining_args = _CallAndUpdateTrace(
                                ^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/fire/core.py", line 684, in _CallAndUpdateTrace
    component = fn(*varargs, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/src/vidkompy/vidkompy.py", line 157, in main
    alignment.process(
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/src/vidkompy/core/alignment_engine.py", line 138, in process
    temporal_alignment = self._compute_temporal_alignment(
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/src/vidkompy/core/alignment_engine.py", line 277, in _compute_temporal_alignment
    return self.temporal_aligner.align_frames(bg_info, fg_info, trim)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/src/vidkompy/core/temporal_alignment.py", line 169, in align_frames
    return self._align_frames_tunnel(bg_info, fg_info, trim)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/src/vidkompy/core/temporal_alignment.py", line 280, in _align_frames_tunnel
    frame_alignments, confidence = self.tunnel_aligner.align(
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/src/vidkompy/core/tunnel_aligner.py", line 354, in align
    return super().align(fg_frames, bg_frames, x_offset, y_offset, verbose)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/src/vidkompy/core/tunnel_aligner.py", line 112, in align
    alignments.append(FrameAlignment(
                      ^^^^^^^^^^^^^^^
TypeError: FrameAlignment.__init__() got an unexpected keyword argument 'fg_frame'

```