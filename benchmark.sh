#!/usr/bin/env bash
#for engine in fast precise; do

for engine in mask; do
    for drift in 16 128; do
        for window in 8 128; do
            base="p-${engine}-d${drift}-w${window}"
            echo ${base}
            time python -m vidkompy -b tests/bg.mp4 -f tests/fg.mp4 -e ${engine} -d ${drift} -w ${window} -o tests/${base}.mp4 --verbose >tests/${base}.txt 2>tests/${base}.err.txt
        done
    done
done
