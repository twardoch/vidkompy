#!/usr/bin/env bash
#for engine in fast precise; do

for engine in fast precise mask; do
    for drift in 8 100; do
        for window in 8 100; do
            base="q-${engine}-d${drift}-w${window}"
            echo ${base}
            time python -m vidkompy -b tests/bg.mp4 -f tests/fg.mp4 -e ${engine} -d ${drift} -w ${window} -o tests/${base}.mp4 --verbose >tests/${base}.txt 2>tests/${base}.err.txt
        done
    done
done
