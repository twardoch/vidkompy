# TODO

- [ ] Test the `precise` engine with the new drift correction (polynomial model, adaptive blend) and Savitzky-Golay smoothing to verify if "flag wave drifts" are eliminated or significantly reduced. Use `benchmark.sh` or a similar script, varying `-d` (drift_interval) and the new parameters if exposed via CLI (or by modifying `PreciseEngineConfig` defaults).

- [ ] Analyze the effectiveness of the `-w` (window) CLI parameter for the `precise` engine. If confirmed it's not being used effectively (as noted in `SPEC.md`), implement a fix so that `DTWAligner` instances within the precise engine respect this parameter where appropriate.

- [ ] Based on testing of Idea 1, decide whether to:
    - Further tune parameters of Idea 1.
    - Proceed to implement Idea 2 (Optical Flow-Assisted Consistency) from `SPEC.md`.
    - Proceed to implement Idea 3 (Dominant Path DTW with Iterative Refinement) from `SPEC.md`.

- [ ] Add unit tests for the new drift correction models and smoothing functions in `MultiResolutionAligner`.

```
#!/usr/bin/env bash
for engine in fast precise; do
    for drift in 1 16 32 64 128; do
        for window in 1 16 32 64 128; do
            base="o-${engine}-d${drift}-w${window}"
            echo ${base}
            time python -m vidkompy -b tests/bg.mp4 -f tests/fg.mp4 -e ${engine} -d ${drift} -w ${window} -o tests/${base}.mp4 --verbose >tests/${base}.txt 2>tests/${base}.err.txt
        done
    done
done
```

The logs are in `tests/` (like `tests/o-fast-d1-w1.err.txt`)

- Between the fast and precise engines, the fast engine is actually better. The precise engine has weird "flag wave drifts", that is the lower part (the bg) goes faster and then slower. 
- The various differences in `-d` and `-w` parameters don't see to make a difference. 

So: I'm happy with how the fast engine is working. 

But I'm not happy with the precise engine. Really, really, there must be a way to temporally aligh the fg frames to the cropped bg frames which is better than what we have now. It can be SLOWER, I don't care (if I want fast, I have the fast engine). 

Analyze the entire codebase. 

Then in `