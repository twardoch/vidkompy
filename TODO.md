# Situation

`src/vidkompy/align/` is a new, modern code which can read a "bg" (background) video or frame, and a "fg" (foreground) video or frame, and find the best spatial alignment (shift and optionally scale) of `fg` within the canvas of `bg`. The --unity_scale option is a boolean flag that, if True, will bias the search towards 100% scale (shift only) detection.

`src/vidkompy/comp/` is the the code that's responsive for actually overlaying a `fg` video onto `bg` video. The code 

- spatially aligns the `fg` within the `bg` canvas
- then also temporally aligns all frames of `bg` to the frames of `fg`
- then actually overlays the `fg` onto the `bg`

## [ ] Task 1

The spatial alignment code in `src/vidkompy/comp/` is different from the spatial alignment code in `src/vidkompy/align/`. We prefer the alignment code in `src/vidkompy/align/`. 

In `PLAN.md`, write a very detailed plan for how to replace the spatial alignment in `src/vidkompy/comp/` with calls to the "unity-scale"  code from `src/vidkompy/align/`.

## [ ] Task 2

Implement the code from `PLAN.md`.

## [ ] Task 3

Update the README.md to reflect the new structure of the code.

## [ ] Task 4

Update the CHANGELOG.md to reflect the new features and changes.
