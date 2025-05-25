# TODO

```
time python -m vidkompy -b tests/bg.mp4 -f tests/fg.mp4 -o tests/o-fast.mp4 -m 1 -e fast;

14:29:36 | INFO     | ✓ Perceptual hashing enabled (pHash)
14:29:36 | INFO     | Analyzing videos...
14:29:36 | INFO     | Video info for bg.mp4: 1920x1080, 60.00 fps, 7.85s, 472 frames, audio: yes
14:29:36 | INFO     | Video info for fg.mp4: 1920x870, 60.89 fps, 8.04s, 483 frames, audio: yes
14:29:36 | INFO     | Video compatibility check:
14:29:36 | INFO     |   Resolution: 1920x1080 vs 1920x870
14:29:36 | INFO     |   FPS: 60.00 vs 60.89
14:29:36 | INFO     |   Duration: 7.85s vs 8.04s
14:29:36 | INFO     |   Audio: yes vs yes
14:29:36 | INFO     | Computing spatial alignment...
14:29:37 | INFO     | Template match found at (0, 0) with confidence 0.941
14:29:37 | INFO     | Spatial alignment result: offset=(0, 0), scale=1.000, confidence=0.941
14:29:37 | INFO     | Computing temporal alignment...
14:29:37 | INFO     | Starting frame-based temporal alignment
14:29:37 | INFO     | Adaptive calculation suggests 50 keyframes
14:29:37 | INFO     | Using 1 keyframes (clamped by max_keyframes=1)
14:29:37 | INFO     | Perceptual hashing mode active. Target keyframes: 1
14:29:37 | INFO     | Sampling every 483 frames for keyframe matching
14:29:37 | INFO     | Sampling 2 FG frames and 2 BG frames
14:29:37 | INFO     | Pre-computing perceptual hashes...
14:29:37 | INFO     | Building cost matrix for dynamic programming alignment...
14:29:37 | WARNING  | Only found 0 matches, attempting refinement...
14:29:37 | INFO     | Found 0 monotonic keyframe matches
14:29:37 | INFO     | ✓ Perceptual hashing provided significant speedup
14:29:37 | WARNING  | No keyframe matches found, using direct mapping
14:29:37 | INFO     | Temporal alignment result: method=direct, offset=0.000s, frames=483, confidence=0.300
14:29:37 | INFO     | Composing output video...
14:29:37 | INFO     | Composing video with direct temporal alignment
Composing frames ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% (483/483 frames) 0:00:00
14:29:49 | INFO     | Wrote 483 frames to /var/folders/05/clcynl0509ldxltl599hhhx40000gn/T/tmpf78v3lyy/temp_silent.mp4
14:29:49 | INFO     | Using foreground audio track
14:29:50 | INFO     | ✅ Processing complete: tests/o-fast.mp4

real	0m15.762s
user	0m21.627s
sys	0m2.923s
```

```
time python -m vidkompy -b tests/bg.mp4 -f tests/fg.mp4 -o tests/o-precise.mp4 -m 1 -e precise;

14:29:52 | INFO     | ✓ Perceptual hashing enabled (pHash)
14:29:52 | INFO     | Analyzing videos...
14:29:53 | INFO     | Video info for bg.mp4: 1920x1080, 60.00 fps, 7.85s, 472 frames, audio: yes
14:29:53 | INFO     | Video info for fg.mp4: 1920x870, 60.89 fps, 8.04s, 483 frames, audio: yes
14:29:53 | INFO     | Video compatibility check:
14:29:53 | INFO     |   Resolution: 1920x1080 vs 1920x870
14:29:53 | INFO     |   FPS: 60.00 vs 60.89
14:29:53 | INFO     |   Duration: 7.85s vs 8.04s
14:29:53 | INFO     |   Audio: yes vs yes
14:29:53 | INFO     | Computing spatial alignment...
14:29:53 | INFO     | Template match found at (0, 0) with confidence 0.941
14:29:53 | INFO     | Spatial alignment result: offset=(0, 0), scale=1.000, confidence=0.941
14:29:53 | INFO     | Computing temporal alignment...
14:29:53 | INFO     | Starting frame-based temporal alignment
14:29:53 | INFO     | Using precise temporal alignment engine
14:29:53 | INFO     | Initialized 4 hash algorithms
14:29:53 | INFO     | Multi-resolution levels: [16, 8, 4, 2]
14:29:53 | INFO     | Extracting frames for precise alignment...
14:29:54 | INFO     | Extracting all 472 frames at 480x270
14:29:56 | INFO     | Extracted 472 frames from bg.mp4
14:29:56 | INFO     | Extracting all 483 frames at 480x217
14:29:59 | INFO     | Extracted 483 frames from fg.mp4
14:29:59 | INFO     | Starting precise temporal alignment
14:29:59 | INFO     | FG: 483 frames, BG: 472 frames
14:29:59 | INFO     | Computing frame fingerprints...
14:29:59 | ERROR    | Precise alignment failed: 'FrameFingerprinter' object has no attribute 'compute_fingerprints'
14:29:59 | INFO     | Falling back to standard alignment
14:29:59 | INFO     | Using DTW-based temporal alignment
14:29:59 | INFO     | DTW sampling: 483 FG frames, 472 BG frames
14:29:59 | INFO     | Computing frame fingerprints...
14:29:59 | INFO     | Computing fingerprints for 483 frames...
14:30:46 | INFO     | Computed 483 fingerprints in 47.41s (10.2 fps)
14:30:46 | INFO     | Computing fingerprints for 472 frames...
14:34:37 | INFO     | Computed 472 fingerprints in 231.20s (2.0 fps)
14:34:37 | INFO     | Running DTW alignment...
14:34:37 | INFO     | Starting DTW alignment: 483 fg frames × 472 bg frames, window=100
14:34:42 | INFO     | DTW completed: 483 frame alignments
14:34:42 | INFO     | DTW alignment complete: 483 frames, offset=0.000s, confidence=-2.574
14:34:42 | INFO     | Temporal alignment result: method=dtw, offset=0.000s, frames=483, confidence=-2.574
14:34:42 | INFO     | Composing output video...
14:34:42 | INFO     | Composing video with dtw temporal alignment
Composing frames ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% (483/483 frames) 0:00:00
14:35:07 | INFO     | Wrote 483 frames to /var/folders/05/clcynl0509ldxltl599hhhx40000gn/T/tmpml3u5lmi/temp_silent.mp4
14:35:07 | INFO     | Using foreground audio track
14:35:08 | INFO     | ✅ Processing complete: tests/o-precise.mp4

real	5m18.183s
user	23m17.843s
sys	2m2.719s
```

1. Both engines are working. 

2. The fast engine is much faster. 

3. Both engines have SOME drift. The precise engine has less drift than the fast engine at the END of the clip, but it actually has more drift at around 1 second mark. 

TASK: Update the README.md file to reflect the new engines. Explain them in HIGH DETAIL! 