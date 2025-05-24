# TODO: 

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
python -m vidkompy --bg tests/bg.mp4 --fg tests/fg.mp4 -o tests/output_new.mp4 --verbose
```

The code is slow, even though we have just an 8 seconds video with just 480 frames. See log below. 

Read `varia/SPEC2.md` for a ideas that can improve the current code.

The resulting `tests/output_new.mp4` is NOT GOOD AT ALL. The fg video portion (the bottom one) is extremely sped up. Below the first second the whole principal movement flies by, and then for the remaining 7 seconds the fg video is almost fully a static image resembling the final frame. 

This is bad. Also the log below suggests that the matching between bg and fg is kind of jumping around. But we DO KNOW that both bg video and fg video always show similar CONTINUOUS action. One can be slightly faster or slower, longer or shorter, but the mapping from bg frams to fg frames should always be unidirectional = going forward. That is, once a given frame `bg_i` has been matched to the frame `fg_i`, then the matching to `fg_i+1` must be of a frame `bg_j` where most certainly is `j >= i` but never `j < i`. 

We should also ensure that the matching is not so that the first 10% of the bg is used quickly and then the rest of the bg is used slowly. In general once we’ve found a good pair of `fg_match_start` and `bg_match_start` and we’ve found a good pair of `fg_match_end` and `bg_match_end`, then matching to the frames between `fg_match_start` and `fg_match_end` should be done so that the frames from `bg_match_start` to `bg_match_end` map monotonically to the fg frames. Remember that `fg_match_start` can be >= `fg_0`, `bg_match_start` can be >= `bg_0`, `fg_match_end` can be <= `fg_last`, and `bg_match_end` can be <= `bg_last`. 

The next step is that you write `varia/SPEC3.md` and in there write a VERY DETAILED PLAN of how you address this problem. 

---

```
19:38:16 | INFO     | __init__ - ✓ Perceptual hashing enabled (pHash)
⠋ Analyzing videos...19:38:16 | DEBUG    | get_video_info - Probing video: tests/bg.mp4
⠸ Analyzing videos...19:38:16 | INFO     | get_video_info - Video info for bg.mp4: 1920x1080, 60.00 fps, 7.85s, 472 frames, audio: yes
19:38:16 | DEBUG    | get_video_info - Probing video: tests/fg.mp4
⠦ Analyzing videos...19:38:16 | INFO     | get_video_info - Video info for fg.mp4: 1920x870, 60.89 fps, 8.04s, 483 frames, audio: yes
19:38:16 | INFO     | _log_compatibility - Video compatibility check:
19:38:16 | INFO     | _log_compatibility -   Resolution: 1920x1080 vs 1920x870
19:38:16 | INFO     | _log_compatibility -   FPS: 60.00 vs 60.89
19:38:16 | INFO     | _log_compatibility -   Duration: 7.85s vs 8.04s
19:38:16 | INFO     | _log_compatibility -   Audio: yes vs yes
⠴ Analyzing videos...           
⠦ Analyzing videos...           
⠦ Computing spatial alignment...19:38:17 | INFO     | _template_matching - Template match found at (0, 0) with confidence 0.9⠧ Analyzing videos...            
⠧ Computing spatial alignment... 
⠧ Computing temporal alignment...19:38:17 | INFO     | align_frames - Starting frame-based temporal alignment
19:38:17 | INFO     | _find_keyframe_matches - Sampling every 1 frames for keyframe matching
⠙ Analyzing videos...            
⠙ Computing spatial alignment... 
⠇ Analyzing videos...            
⠴ Analyzing videos...            
⠴ Computing spatial alignment... 
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[250] for fg[1]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[250] for fg[2]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[0] for fg[3]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[250] for fg[4]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[3] for fg[5]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[3] for fg[6]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[3] for fg[7]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[3] for fg[8]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[3] for fg[9]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[3] for fg[10]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[3] for fg[11]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[3] for fg[12]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[3] for fg[13]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[3] for fg[14]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[16] for fg[15]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[13] for fg[16]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[30] for fg[17]
(...)
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[16] for fg[25]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[16] for fg[26]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[16] for fg[28]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[16] for fg[29]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[448] for fg[30]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[448] for fg[31]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[16] for fg[32]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[47] for fg[33]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[297] for fg[34]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[47] for fg[35]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[448] for fg[36]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[16] for fg[37]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[448] for fg[38]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[16] for fg[39]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[448] for fg[40]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[448] for fg[41]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[448] for fg[42]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[250] for fg[43]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[448] for fg[44]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[448] for fg[45]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[448] for fg[46]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[448] for fg[47]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[250] for fg[48]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[444] for fg[49]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[250] for fg[50]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[95] for fg[51]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[95] for fg[53]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[95] for fg[54]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[47] for fg[72]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[47] for fg[74]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[47] for fg[79]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[47] for fg[81]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[47] for fg[82]
(...)
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[450] for fg[475]
19:45:40 | DEBUG    | _filter_monotonic - Filtering non-monotonic match: bg[450] for fg[477]
19:45:40 | INFO     | _find_keyframe_matches - Found 5 keyframe matches in 0.68s (709.5 fps)
⠧ Analyzing videos...            
⠧ Analyzing videos...            
⠸ Analyzing videos...            
⠼ Analyzing videos...            
⠏ Analyzing videos...            
⠧ Analyzing videos...            
⠼ Analyzing videos...            
⠼ Computing spatial alignment... 
⠏ Analyzing videos...            
⠋ Analyzing videos...            
⠋ Computing spatial alignment... 
⠋ Computing temporal alignment...
⠋ Composing output video...      
```