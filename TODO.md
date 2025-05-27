# TODO

Read `varia/thumbfind1.md`, `varia/thumbfind2.md`, `varia/thumbfind3.md`

Then think about how to improve `src/vidkompy/thumbfind.py`: 



Can't we use a much faster algorithm to find the ballpark match for a scale, and then refine very precisely? Ultimately I do want to be as close as possible. We could print the gradual matching numbers, I mean perhaps instead of --precise we need a CLI switch --precision which would take 1 or 2 or 3 or 4 etc. After each precision step we’d display results, and each step could principally use different tech. 

```
python -m vidkompy find vlveo-toronto-aisha2.mp4 vlveo-toronto-aisha3_starlight.mp4 -n 9 -p -v
```

```
Thumbnail Finder
Foreground: vlveo-toronto-aisha2.mp4
Background: vlveo-toronto-aisha3_starlight.mp4
Max frames: 7
Precise mode enabled - may take longer but provides higher accuracy
Method: AUTO

Extracting frames...
Extracted 7 foreground frames
Extracted 7 background frames

Searching for thumbnails... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% -:--:--DEBUG: AKAZE: scale=1.005, pos=(-3.8,104.1), conf=0.875, inliers=56/64
DEBUG: Feature matching result: Scale: 100.51%, Position: (-3, 104), Confidence: 0.875
DEBUG: Testing 20 scales from 0.81 to 1.00 using parallel processing
Searching for thumbnails... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% -:--:--DEBUG: Parallel template matching: scale=0.990, pos=(9,114), correlation=0.970
DEBUG: Template matching result: Scale: 98.97%, Position: (9, 114), Confidence: 0.970
Searching for thumbnails... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% -:--:--DEBUG: Phase correlation: shift=(4.30, 110.70), confidence=0.000
DEBUG: Phase correlation result: Scale: 98.97%, Position: (4, 110), Confidence: 0.000
DEBUG: Hybrid matching final result: Scale: 98.97%, Position: (9, 114), Confidence: 0.970
DEBUG: FG frame 0 vs BG frame 0: Scale: 98.97%, Position: (9, 114), Confidence: 0.970
Searching for thumbnails... ╸

(...)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺  98% 0:00:04DEBUG: Parallel template matching: scale=0.982, pos=(27,115), correlation=0.900
DEBUG: Template matching result: Scale: 98.16%, Position: (27, 115), Confidence: 0.900
Searching for thumbnails... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺  98% 0:00:04DEBUG: Phase correlation: shift=(-2.30, 135.00), confidence=0.000
DEBUG: Phase correlation result: Scale: 98.16%, Position: (-2, 135), Confidence: 0.000
DEBUG: Hybrid matching final result: Scale: 98.16%, Position: (27, 115), Confidence: 0.900
DEBUG: FG frame 6 vs BG frame 6: Scale: 98.16%, Position: (27, 115), Confidence: 0.900
Searching for thumbnails... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
DEBUG: Preferring unity scale result: confidence=0.978 vs scaled=0.978
INFO: Applying precise refinement...
DEBUG: Starting precise refinement around scale=0.986, pos=(17,114)
DEBUG: Testing 2541 combinations for precise refinement
Precise refinement... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% -:--:--DEBUG: bytecode dump:
>          0	NOP(arg=None, lineno=168)
           2	RESUME(arg=0, lineno=168)
           4	LOAD_GLOBAL(arg=1, lineno=174)
          14	LOAD_ATTR(arg=2, lineno=174)
          34	LOAD_FAST(arg=0, lineno=174)
          36	CALL(arg=1, lineno=174)
          44	STORE_FAST(arg=2, lineno=174)
          46	LOAD_GLOBAL(arg=1, lineno=175)
          56	LOAD_ATTR(arg=2, lineno=175)
          76	LOAD_FAST(arg=1, lineno=175)
          78	CALL(arg=1, lineno=175)
          86	STORE_FAST(arg=3, lineno=175)
          88	LOAD_GLOBAL(arg=1, lineno=177)
          98	LOAD_ATTR(arg=4, lineno=177)
         118	LOAD_FAST(arg=0, lineno=177)
         120	LOAD_FAST(arg=2, lineno=177)
         122	BINARY_OP(arg=10, lineno=177)
         126	LOAD_FAST(arg=1, lineno=177)
         128	LOAD_FAST(arg=3, lineno=177)
         130	BINARY_OP(arg=10, lineno=177)
         134	BINARY_OP(arg=5, lineno=177)
         138	CALL(arg=1, lineno=177)
         146	STORE_FAST(arg=4, lineno=177)
         148	LOAD_GLOBAL(arg=1, lineno=178)
         158	LOAD_ATTR(arg=4, lineno=178)
         178	LOAD_FAST(arg=0, lineno=178)
         180	LOAD_FAST(arg=2, lineno=178)
         182	BINARY_OP(arg=10, lineno=178)
         186	LOAD_CONST(arg=1, lineno=178)
         188	BINARY_OP(arg=8, lineno=178)
         192	CALL(arg=1, lineno=178)
         200	STORE_FAST(arg=5, lineno=178)
         202	LOAD_GLOBAL(arg=1, lineno=179)
         212	LOAD_ATTR(arg=4, lineno=179)
         232	LOAD_FAST(arg=1, lineno=179)
         234	LOAD_FAST(arg=3, lineno=179)
         236	BINARY_OP(arg=10, lineno=179)
         240	LOAD_CONST(arg=1, lineno=179)
         242	BINARY_OP(arg=8, lineno=179)
         246	CALL(arg=1, lineno=179)
         254	STORE_FAST(arg=6, lineno=179)
         256	LOAD_FAST(arg=5, lineno=181)
         258	LOAD_CONST(arg=2, lineno=181)
         260	COMPARE_OP(arg=40, lineno=181)
         264	POP_JUMP_IF_TRUE(arg=5, lineno=181)
         266	LOAD_FAST(arg=6, lineno=181)
         268	LOAD_CONST(arg=2, lineno=181)
         270	COMPARE_OP(arg=40, lineno=181)
         274	POP_JUMP_IF_FALSE(arg=1, lineno=181)
>        276	RETURN_CONST(arg=3, lineno=182)
>        278	LOAD_FAST(arg=4, lineno=184)
         280	LOAD_GLOBAL(arg=1, lineno=184)
         290	LOAD_ATTR(arg=6, lineno=184)
         310	LOAD_FAST(arg=5, lineno=184)
         312	LOAD_FAST(arg=6, lineno=184)
         314	BINARY_OP(arg=5, lineno=184)
         318	CALL(arg=1, lineno=184)
         326	BINARY_OP(arg=11, lineno=184)
         330	RETURN_VALUE(arg=None, lineno=184)
DEBUG: pending: deque([State(pc_initial=0 nstack_initial=0)])
DEBUG: stack: []
DEBUG: state.pc_initial: State(pc_initial=0 nstack_initial=0)
DEBUG: dispatch pc=0, inst=NOP(arg=None, lineno=168)
DEBUG: stack []
DEBUG: dispatch pc=2, inst=RESUME(arg=0, lineno=168)

(...)
```