#!/usr/bin/env bash
# Test all engines

for engine in mask full; do
    for drift in 10 100; do
        for window in 10 100; do
            base="q-${engine}-d${drift}-w${window}"
            echo
            echo ">> ENGINE: ${engine} DRIFT: ${drift} WINDOW: ${window}"
            echo ">> OUTPUT ${base}.mp4"
            time python -m vidkompy -b tests/bg.mp4 -f tests/fg.mp4 -e ${engine} -d ${drift} -w ${window} -m 48 -s -o tests/${base}.mp4 --verbose
        done
    done
done
