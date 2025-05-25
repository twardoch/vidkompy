#!/usr/bin/env bash
# Test all engines including the new tunnel engines

# for engine in tunnel_mask tunnel_full mask precise fast; do
for engine in tunnel_mask tunnel_full; do
    # For tunnel engines, only window parameter matters
    if [[ $engine == tunnel_* ]]; then
        for window in 30 60 100; do
            base="q-${engine}-w${window}"
            echo ${base}
            time python -m vidkompy -b tests/bg.mp4 -f tests/fg.mp4 -e ${engine} -w ${window} -o tests/${base}.mp4 --verbose >tests/${base}.txt 2>tests/${base}.err.txt
        done
    else
        # For other engines, test drift and window combinations
        for drift in 8 100; do
            for window in 8 100; do
                base="q-${engine}-d${drift}-w${window}"
                echo ${base}
                time python -m vidkompy -b tests/bg.mp4 -f tests/fg.mp4 -e ${engine} -d ${drift} -w ${window} -o tests/${base}.mp4 --verbose >tests/${base}.txt 2>tests/${base}.err.txt
            done
        done
    fi
done
