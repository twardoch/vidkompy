# TODO

The tool works OK now, way better than ever before. 

When I run `python -m vidkompy -b tests/bg.mp4 -f tests/fg.mp4 -o tests/test.mp4 -m 1`, we get: 

```
13:34:31 | INFO     | ✓ Perceptual hashing enabled (pHash)
13:34:31 | INFO     | Analyzing videos...
13:34:31 | INFO     | Video info for bg.mp4: 1920x1080, 60.00 fps, 7.85s, 472 frames, audio: yes
13:34:31 | INFO     | Video info for fg.mp4: 1920x870, 60.89 fps, 8.04s, 483 frames, audio: yes
13:34:31 | INFO     | Video compatibility check:
13:34:31 | INFO     |   Resolution: 1920x1080 vs 1920x870
13:34:31 | INFO     |   FPS: 60.00 vs 60.89
13:34:31 | INFO     |   Duration: 7.85s vs 8.04s
13:34:31 | INFO     |   Audio: yes vs yes
13:34:31 | INFO     | Computing spatial alignment...
13:34:32 | INFO     | Template match found at (0, 0) with confidence 0.941
13:34:32 | INFO     | Spatial alignment result: offset=(0, 0), scale=1.000, confidence=0.941
13:34:32 | INFO     | Computing temporal alignment...
13:34:32 | INFO     | Starting frame-based temporal alignment
13:34:32 | INFO     | Adaptive calculation suggests 50 keyframes
13:34:32 | INFO     | Using 1 keyframes (clamped by max_keyframes=1)
13:34:32 | INFO     | Perceptual hashing mode active. Target keyframes: 1
13:34:32 | INFO     | Sampling every 483 frames for keyframe matching
13:34:32 | INFO     | Sampling 2 FG frames and 2 BG frames
13:34:32 | INFO     | Pre-computing perceptual hashes...
13:34:32 | INFO     | Building cost matrix for dynamic programming alignment...
13:34:32 | WARNING  | Only found 0 matches, attempting refinement...
13:34:32 | INFO     | Found 0 monotonic keyframe matches
13:34:32 | INFO     | ✓ Perceptual hashing provided significant speedup
13:34:32 | WARNING  | No keyframe matches found, using direct mapping
13:34:32 | INFO     | Temporal alignment result: method=direct, offset=0.000s, frames=483, confidence=0.300
13:34:32 | INFO     | Composing output video...
13:34:32 | INFO     | Composing video with direct temporal alignment
Composing frames ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% (483/483 frames) 0:00:00
13:34:41 | INFO     | Wrote 483 frames to /var/folders/05/clcynl0509ldxltl599hhhx40000gn/T/tmpmrnzk6ox/temp_silent.mp4
13:34:41 | INFO     | Using foreground audio track
13:34:41 | INFO     | ✅ Processing complete: tests/test.mp4
```

The result still has some drift. 

Analyze the entire codebase. It’s good to keep the current implementation, let’s add a CLI param `--engine` and the current implementation is the `fast` engine. 

Let’s implement a 2nd engine, `precise`, which would use a much more detailed temporal alignment. 

First research the best way to do this. 

Then write a `SPEC.md` file that describes the new engine. 