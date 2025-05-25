# TODO

## FIXED: FrameAlignment attribute error

**Issue:** The tunnel_mask engine was trying to access `bg_frame` and `fg_frame` attributes on `FrameAlignment` objects, but the correct attribute names are `bg_frame_idx` and `fg_frame_idx`.

**Fix Applied:** Updated `temporal_alignment.py` line 291 to use the correct attribute names:
- Changed `first_align.bg_frame` → `first_align.bg_frame_idx`  
- Changed `first_align.fg_frame` → `first_align.fg_frame_idx`

**Status:** Ready for testing with `./benchmark.sh`