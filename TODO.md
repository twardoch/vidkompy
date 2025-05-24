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
time python -m vidkompy  --bg tests/bg.mp4 --fg tests/fg.mp4 -o tests/out-2.mp4 --verbose --blend --border 20 --window 4
```

# TODO

There is still visible drift despite the use of the border mode with a windows = 4 frames. 

