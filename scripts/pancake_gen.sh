#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR/../src
cd $SCRIPT_DIR/../

usage() {
    echo "Usage: $0 [d|12|16|32|64]"
    exit 1
}
if [ $# -ne 1 ]; then
    usage
fi

if [ "$1" = "d" ]; then
    python src/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  problems/pancake16d/ \
    --seed 658 \
    --width  16 \
    --n-problems-per-stage  25 \
    --randomize-curriculum-steps \
    --stages-multiple  1 \
    --n-problems-final-stage 2500 \
    --n-stages  101 \
    --n-valid  100 \
    --n-test 100 \
    --randomize-test-steps \
    --test-steps  100
elif [ "$1" = "12" ]; then
    python src/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  problems/pancake12/ \
    --seed 658 \
    --width  12 \
    --n-problems-per-stage  250 \
    --randomize-curriculum-steps \
    --stages-multiple  1 \
    --n-problems-final-stage 25000 \
    --n-stages  101 \
    --n-valid  1000 \
    --n-test 1000 \
    --randomize-test-steps \
    --test-steps  100
elif [ "$1" = "16" ]; then
    python src/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  problems/pancake16/ \
    --seed 358 \
    --width  16 \
    --n-problems-per-stage  250 \
    --randomize-curriculum-steps \
    --stages-multiple  1 \
    --n-problems-final-stage 25000 \
    --n-stages  101 \
    --n-valid  1000 \
    --n-test 1000 \
    --randomize-test-steps \
    --test-steps  100
elif [ "$1" = "32" ]; then
    python src/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  problems/pancake32/ \
    --seed 3721 \
    --width  32 \
    --n-problems-per-stage  250 \
    --randomize-curriculum-steps \
    --stages-multiple  1 \
    --n-problems-final-stage 25000 \
    --n-stages  101 \
    --n-valid  1000 \
    --n-test 1000 \
    --randomize-test-steps \
    --test-steps  100
elif [ "$1" = "64" ]; then
    python src/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  problems/pancake64/ \
    --seed 3358 \
    --width  64 \
    --n-problems-per-stage  250 \
    --randomize-curriculum-steps \
    --stages-multiple  1 \
    --n-problems-final-stage 25000 \
    --n-stages  101 \
    --n-valid  1000 \
    --n-test 1000 \
    --randomize-test-steps \
    --test-steps  100
else
    usage
fi
