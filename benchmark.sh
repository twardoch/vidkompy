#!/usr/bin/env bash
# Test default engine (Full) for MVP

# For MVP, engine is fixed to "full". The -e flag is removed.
# The loop for 'engine' can be removed or fixed to one iteration.
# Let's fix it to one conceptual "default_mvp" run.
engine_mvp_name="full_default_mvp"

for drift in 10 100; do
    for window in 10 100; do
        base="q-${engine_mvp_name}-d${drift}-w${window}" # Use new conceptual name for output file
        echo
        echo ">> ENGINE: ${engine_mvp_name} DRIFT: ${drift} WINDOW: ${window}"
        echo ">> OUTPUT ${base}.mp4"
        # Removed -e ${engine} from the command
        time python -m vidkompy comp -b tests/bg.mp4 -f tests/fg.mp4 -d ${drift} -w ${window} -m 48 --unscaled true --align_precision 2 -o tests/${base}.mp4 --verbose
    done
done
