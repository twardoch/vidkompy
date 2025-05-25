# TODO

## Experiment

`time python -m vidkompy -b tests/bg.mp4 -f tests/fg.mp4 -o tests/o-fast1.mp4 -m 1 -e fast; time python -m vidkompy -b tests/bg.mp4 -f tests/fg.mp4 -o tests/o-precise1.mp4 -m 1 -e precise;`

Here are the results: 

```
time python -m vidkompy -b tests/bg.mp4 -f tests/fg.mp4 -o tests/o-fast1.mp4 -m 1 -e fast;

15:06:29 | INFO     | ✓ Perceptual hashing enabled (pHash)
15:06:29 | INFO     | Analyzing videos...
15:06:30 | INFO     | Video info for bg.mp4: 1920x1080, 60.00 fps, 7.85s, 472 frames, audio: yes
15:06:31 | INFO     | Video info for fg.mp4: 1920x870, 60.89 fps, 8.04s, 483 frames, audio: yes
15:06:31 | INFO     | Video compatibility check:
15:06:31 | INFO     |   Resolution: 1920x1080 vs 1920x870
15:06:31 | INFO     |   FPS: 60.00 vs 60.89
15:06:31 | INFO     |   Duration: 7.85s vs 8.04s
15:06:31 | INFO     |   Audio: yes vs yes
15:06:31 | INFO     | Computing spatial alignment...
15:06:32 | INFO     | Template match found at (0, 0) with confidence 0.941
15:06:32 | INFO     | Spatial alignment result: offset=(0, 0), scale=1.000, confidence=0.941
15:06:32 | INFO     | Computing temporal alignment...
15:06:32 | INFO     | Starting frame-based temporal alignment
15:06:32 | INFO     | Adaptive calculation suggests 50 keyframes
15:06:32 | INFO     | Using 1 keyframes (clamped by max_keyframes=1)
15:06:32 | INFO     | Perceptual hashing mode active. Target keyframes: 1
15:06:32 | INFO     | Sampling every 483 frames for keyframe matching
15:06:32 | INFO     | Sampling 2 FG frames and 2 BG frames
15:06:32 | INFO     | Pre-computing perceptual hashes...
15:06:33 | INFO     | Building cost matrix for dynamic programming alignment...
15:06:33 | WARNING  | Only found 0 matches, attempting refinement...
15:06:33 | INFO     | Found 0 monotonic keyframe matches
15:06:33 | INFO     | ✓ Perceptual hashing provided significant speedup
15:06:33 | WARNING  | No keyframe matches found, using direct mapping
15:06:33 | INFO     | Temporal alignment result: method=direct, offset=0.000s, frames=483, confidence=0.300
15:06:33 | INFO     | Composing output video...
15:06:33 | INFO     | Composing video with direct temporal alignment
Composing frames ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% (483/483 frames) 0:00:00
15:06:53 | INFO     | Wrote 483 frames to /var/folders/05/clcynl0509ldxltl599hhhx40000gn/T/tmpbg24brjd/temp_silent.mp4
15:06:53 | INFO     | Using foreground audio track
15:06:54 | INFO     | ✅ Processing complete: tests/o-fast1.mp4

real	0m48.269s
user	0m42.203s
sys	0m5.009s
```

It starts very well, but around 05:00 it starts to drift. 

```
time python -m vidkompy -b tests/bg.mp4 -f tests/fg.mp4 -o tests/o-precise1.mp4 -m 1 -e precise;

15:06:58 | INFO     | ✓ Perceptual hashing enabled (pHash)
15:06:58 | INFO     | Analyzing videos...
15:06:59 | INFO     | Video info for bg.mp4: 1920x1080, 60.00 fps, 7.85s, 472 frames, audio: yes
15:06:59 | INFO     | Video info for fg.mp4: 1920x870, 60.89 fps, 8.04s, 483 frames, audio: yes
15:06:59 | INFO     | Video compatibility check:
15:06:59 | INFO     |   Resolution: 1920x1080 vs 1920x870
15:06:59 | INFO     |   FPS: 60.00 vs 60.89
15:06:59 | INFO     |   Duration: 7.85s vs 8.04s
15:06:59 | INFO     |   Audio: yes vs yes
15:06:59 | INFO     | Computing spatial alignment...
15:07:00 | INFO     | Template match found at (0, 0) with confidence 0.941
15:07:00 | INFO     | Spatial alignment result: offset=(0, 0), scale=1.000, confidence=0.941
15:07:00 | INFO     | Computing temporal alignment...
15:07:00 | INFO     | Starting frame-based temporal alignment
15:07:00 | INFO     | Using precise temporal alignment engine
15:07:00 | INFO     | Initialized 4 hash algorithms
15:07:00 | INFO     | Multi-resolution levels: [16, 8, 4, 2]
15:07:00 | INFO     | Extracting frames for precise alignment...
15:07:00 | INFO     | Extracting all 472 frames at 480x270
15:07:04 | INFO     | Extracted 472 frames from bg.mp4
15:07:04 | INFO     | Extracting all 483 frames at 480x217
15:07:06 | INFO     | Extracted 483 frames from fg.mp4
15:07:06 | INFO     | Starting precise temporal alignment
15:07:06 | INFO     | FG: 483 frames, BG: 472 frames
15:07:06 | INFO     | Computing frame fingerprints...
15:07:06 | ERROR    | Precise alignment failed: 'FrameFingerprinter' object has no attribute 'compute_fingerprints'
15:07:06 | INFO     | Falling back to standard alignment
15:07:06 | INFO     | Using DTW-based temporal alignment
15:07:06 | INFO     | DTW sampling: 483 FG frames, 472 BG frames
15:07:06 | INFO     | Computing frame fingerprints...
15:07:06 | INFO     | Computing fingerprints for 483 frames...
15:07:49 | INFO     | Computed 483 fingerprints in 42.57s (11.3 fps)
15:07:49 | INFO     | Computing fingerprints for 472 frames...
15:09:55 | INFO     | Computed 472 fingerprints in 125.94s (3.7 fps)
15:09:55 | INFO     | Running DTW alignment...
15:09:55 | INFO     | Starting DTW alignment: 483 fg frames × 472 bg frames, window=100
15:09:58 | INFO     | DTW completed: 483 frame alignments
15:09:58 | INFO     | DTW alignment complete: 483 frames, offset=0.000s, confidence=-2.574
15:09:58 | INFO     | Temporal alignment result: method=dtw, offset=0.000s, frames=483, confidence=-2.574
15:09:58 | INFO     | Composing output video...
15:09:58 | INFO     | Composing video with dtw temporal alignment
Composing frames ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% (483/483 frames) 0:00:00
15:10:16 | INFO     | Wrote 483 frames to /var/folders/05/clcynl0509ldxltl599hhhx40000gn/T/tmpkn3ui_85/temp_silent.mp4
15:10:17 | INFO     | Using foreground audio track
15:10:17 | INFO     | ✅ Processing complete: tests/o-precise1.mp4

real	3m23.217s
user	16m18.655s
sys	1m13.716s
```

The drift is overall much smaller, but it’s still there. 

## Results

1. Both engines are working. 

2. The fast engine is much faster. 

3. Both engines have SOME drift. The precise engine has less drift than the fast engine, but still has drift. 

4. We have `15:07:06 | ERROR    | Precise alignment failed: 'FrameFingerprinter' object has no attribute 'compute_fingerprints'`

TASK: Fix the error, reduce drift even further in the precise engine. Read `SPEC.md` for ideas.  