- Always assume that the fg video is the "better quality" video, and never should be re-timed 
- Use `tests/bg.mp4` and `tests/fg.mp4` for testing. 
- Work in rounds
- Create `PROGRESS.md` as a detailed flat plan with `[ ]` items. 
- Identify the most important TODO items, and create `TODO.md` with `[ ]` items. 
- Implement the changes. 
- Update `PROGRESS.md` and `TODO.md` as you go. 
- After each round of changes, update `CHANGELOG.md` with the changes.
- Update `README.md` to reflect the changes.

```
time python -m vidkompy --bg tests/bg.mp4 --fg tests/fg.mp4 -o tests/out-b9.mp4 --verbose --blend --border 20 --window 4 --max_keyframes 12

01:13:57 | INFO     | __init__ - ✓ Perceptual hashing enabled (pHash)
01:13:57 | INFO     | process - Analyzing videos...
01:13:57 | DEBUG    | get_video_info - Probing video: tests/bg.mp4
01:13:58 | INFO     | get_video_info - Video info for bg.mp4: 1920x1080, 60.00 fps, 7.85s, 472 frames, audio: yes
01:13:58 | DEBUG    | get_video_info - Probing video: tests/fg.mp4
01:13:58 | INFO     | get_video_info - Video info for fg.mp4: 1920x870, 60.89 fps, 8.04s, 483 frames, audio: yes
01:13:58 | INFO     | _log_compatibility - Video compatibility check:
01:13:58 | INFO     | _log_compatibility -   Resolution: 1920x1080 vs 1920x870
01:13:58 | INFO     | _log_compatibility -   FPS: 60.00 vs 60.89
01:13:58 | INFO     | _log_compatibility -   Duration: 7.85s vs 8.04s
01:13:58 | INFO     | _log_compatibility -   Audio: yes vs yes
01:13:58 | INFO     | process - Computing spatial alignment...
01:13:58 | DEBUG    | _template_matching - Using template matching for spatial alignment
01:13:58 | INFO     | _template_matching - Template match found at (0, 0) with confidence 0.941
01:13:58 | INFO     | process - Spatial alignment result: offset=(0, 0), scale=1.000, confidence=0.941
01:13:58 | INFO     | process - Computing temporal alignment...
01:13:58 | INFO     | _compute_temporal_alignment - Using border-based temporal alignment (border thickness: 20px)
01:13:58 | DEBUG    | create_border_mask - Created border mask: 38400 pixels in border region
01:13:58 | INFO     | align_frames_with_mask - Border mode active for temporal alignment.
01:13:58 | WARNING  | align_frames_with_mask - Border mode forces classic (keyframe/SSIM) alignment to ensure mask is respected, even if DTW is selected. DTW with perceptual hashing does not yet support masked regions effectively.
01:13:58 | INFO     | align_frames_with_mask - In border mode (classic path), forcing SSIM comparison (disabling perceptual hash) to ensure mask is applied.
01:13:58 | INFO     | align_frames_with_mask - Using classic (keyframe-based) temporal alignment. Hash/SSIM: Disabled (SSIM).
01:13:58 | INFO     | _find_keyframe_matches - SSIM mode active for keyframe matching. Target keyframes: 12
01:13:58 | INFO     | _find_keyframe_matches - Sampling every 40 frames for keyframe matching
01:13:58 | INFO     | _find_keyframe_matches - Sampling 14 FG frames and 24 BG frames
01:13:58 | INFO     | _find_keyframe_matches - Building cost matrix for dynamic programming alignment...
01:13:58 | DEBUG    | _build_cost_matrix - Building cost matrix using frame extraction
01:16:55 | INFO     | _find_keyframe_matches - Found 13 monotonic keyframe matches
01:16:55 | INFO     | _build_frame_alignments - Building alignment for FG frames 40 to 482
01:16:55 | INFO     | _build_frame_alignments - Created 443 frame alignments
01:16:55 | INFO     | process - Temporal alignment result: method=border (classic/SSIM), offset=0.010s, frames=443, confidence=0.366
01:16:55 | INFO     | process - Composing output video...
01:16:55 | INFO     | _compose_video - Composing video with border (classic/SSIM) temporal alignment
01:16:57 | DEBUG    | create_blend_mask - Created blend mask with 20px gradient
01:16:57 | INFO     | _compose_with_opencv - Using blend mode with 20px gradient
Composing frames ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% (443/443 frames) 0:00:00
01:17:27 | INFO     | _compose_with_opencv - Wrote 443 frames to /var/folders/05/clcynl0509ldxltl599hhhx40000gn/T/tmpsundewo0/temp_silent.mp4
01:17:27 | INFO     | _add_audio_track - Using foreground audio track
01:17:28 | INFO     | process - ✅ Processing complete: tests/out-b9.mp4

real	3m32.776s
user	9m9.942s
sys	0m55.536s
```

# TODO

<problem>
There is still visible drift. With `--max_keyframes 12` it’s even worse. And the stuff is really slow. 

Step back and analyze the entire codebase. Then propose a detailed SPEC for improvements. 
</problem>

<plan>
In `varia/SPEC5.md`
</plan>

<task>
Implement the plan
</task>