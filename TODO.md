# TODO: 

- The old code is in `src/vidkompy/vidkompy_old.py`.  
- Always assume that the fg video is the "better quality" video, and never should be re-timed 
- Use `tests/bg.mp4` and `tests/fg.mp4` for testing. 

Work in rounds: 

- Create `PROGRESS.md` as a detailed flat plan with `[ ]` items. 
- Identify the most important TODO items, and create `TODO.md` with `[ ]` items. 
- Implement the changes. 
- Update `PROGRESS.md` and `TODO.md` as you go. 
- After each round of changes, update `CHANGELOG.md` with the changes.
- Update `README.md` to reflect the changes.

```
python -m vidkompy --bg tests/bg.mp4 --fg tests/fg.mp4 -o tests/output_new.mp4 --match_time precise --verbose
```

```
⠋ Analyzing videos...17:48:03 | DEBUG    | get_video_info - Probing video: tests/bg.mp4
⠴ Analyzing videos...17:48:03 | INFO     | get_video_info - Video info for bg.mp4: 1920x1080, 60.00 fps, 7.85s, 472 frames, audio: yes
17:48:03 | DEBUG    | get_video_info - Probing video: tests/fg.mp4
⠏ Analyzing videos...17:48:04 | INFO     | get_video_info - Video info for fg.mp4: 1920x870, 60.89 fps, 8.04s, 483 frames, audio: yes
17:48:04 | INFO     | _log_compatibility - Video compatibility check:
17:48:04 | INFO     | _log_compatibility -   Resolution: 1920x1080 vs 1920x870
17:48:04 | INFO     | _log_compatibility -   FPS: 60.00 vs 60.89
17:48:04 | INFO     | _log_compatibility -   Duration: 7.85s vs 8.04s
17:48:04 | INFO     | _log_compatibility -   Audio: yes vs yes
⠏ Analyzing videos...           
⠋ Analyzing videos...           
⠋ Computing spatial alignment...17:48:05 | INFO     | _template_matching - Template match found at (0, 0) with confidence 0.9⠙ Analyzing videos...            
⠙ Computing spatial alignment... 
⠴ Analyzing videos...            
⠴ Computing spatial alignment... 
⠴ Computing temporal alignment...
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/src/vidkompy/__main__.py", line 10, in <module>
    fire.Fire(main)
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/.venv/lib/python3.12/site-packages/fire/core.py", line 135, in Fire
    component_trace = _Fire(component, args, parsed_flag_args, context, name)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/.venv/lib/python3.12/site-packages/fire/core.py", line 468, in _Fire
    component, remaining_args = _CallAndUpdateTrace(
                                ^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/.venv/lib/python3.12/site-packages/fire/core.py", line 684, in _CallAndUpdateTrace
    component = fn(*varargs, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/src/vidkompy/vidkompy.py", line 131, in main
    engine.process(
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/src/vidkompy/core/alignment_engine.py", line 98, in process
    temporal_alignment = self._compute_temporal_alignment(
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/src/vidkompy/core/alignment_engine.py", line 162, in _compute_temporal_alignment
    return self.temporal_aligner.align_frames(bg_info, fg_info, trim)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/src/vidkompy/core/temporal_alignment.py", line 110, in align_frames
    keyframe_matches = self._find_keyframe_matches(bg_info, fg_info)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/src/vidkompy/core/temporal_alignment.py", line 188, in _find_keyframe_matches
    best_match = self._find_best_match(
                 ^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/src/vidkompy/core/temporal_alignment.py", line 238, in _find_best_match
    similarity = self._compute_frame_similarity(bg_frame, fg_frame)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/src/vidkompy/core/temporal_alignment.py", line 263, in _compute_frame_similarity
    score, _ = ssim(gray1, gray2, full=True)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/.venv/lib/python3.12/site-packages/skimage/metrics/_structural_similarity.py", line 248, in structural_similarity
    uy = filter_func(im2, **filter_args)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/.venv/lib/python3.12/site-packages/scipy/ndimage/_filters.py", line 1181, in uniform_filter
    uniform_filter1d(input, int(size), axis, output, mode,
  File "/Users/adam/Developer/vcs/github.twardoch/pub/vidkompy/.venv/lib/python3.12/site-packages/scipy/ndimage/_filters.py", line 1109, in uniform_filter1d
    _nd_image.uniform_filter1d(input, size, axis, output, mode, cval,
KeyboardInterrupt
```