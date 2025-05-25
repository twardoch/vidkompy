# TODO

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
- The various differences in `-d` and `-w` parameters don’t see to make a difference. 

So: I’m happy with how the fast engine is working. 

But I’m not happy with the precise engine. Really, really, there must be a way to temporally aligh the fg frames to the cropped bg frames which is better than what we have now. It can be SLOWER, I don’t care (if I want fast, I have the fast engine). 

Analyze the entire codebase. 

Then in `SPEC.md` write down a detailed documentation that explains how exactly the precise engine is currently working. 

Then in `SPEC.md` write down three ideas for how to improve the temporal alignment precision of the precise engine. 

Then in `SPEC.md` write down a detailed specification that instructs a junior dev to take the steps you will take to improve the temporal alignment precision of the precise engine. 
