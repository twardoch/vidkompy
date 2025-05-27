# TODO.md

## Structure

The code has two parts: 

### Align

Version A of spatial alignment, a tool that is trying to align two frames or two videos in situations where frame/video `fg` is part of frame/video `bg` (that is, the `bg` is an OUTPAINTED variant of `fg`), even if `bg` is downsampled. 

- `src/vidkompy/align_old.py` is the old code
- `src/vidkompy/align/` is the new refactored code

> TASK 1: RIGHT NOW we were in the middle of refactoring `src/vidkompy/align_old.py` into `src/vidkompy/align/`. Finish this refactoring.

### Comp

Version B (simplified) version of spatial alignment, plus temporal alignment of two videos, and actual comping (ovelaying) of `fg` video over `bg` video.

### Common

- `src/vidkompy/__main__.py` is the CLI entry point to the whole package (that is, `src/vidkompy/comp/` and `src/vidkompy/align/`)

> TASK 2: Ensure that `python -m vidkompy align` (formerly `find`) works with the newly refactored `align` code. Also ensure that `python -m vidkompy comp` works with `comp` code.

## Next TODO

- `varia` contains background writing on both portions of the code. 

> TASK 3: Update README.md to reflect the new structure of the code. 

> TASK 4: Update CHANGELOG.md to reflect the new features and changes. 

> TASK 5: Identify the next steps and describe them in detail in `PLAN.md`.

