#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR/../src/bilevin
cd $SCRIPT_DIR/../

usage() {
    echo "Usage: $0 [d|14|18]"
    exit 1
}
if [ $# -ne 1 ]; then
    usage
fi

if [ "$1" = "d" ]; then
    python src/bilevin/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  problems/pancake14d/ \
    --seed 658 \
    --width  14 \
    --n-problems-per-stage  250 \
    --randomize-curriculum-steps \
    --stages-multiple  5 \
    --n-problems-final-stage 500 \
    --n-stages  11 \
    --n-valid  100 \
    --n-test 100 \
    --randomize-test-steps \
    --test-steps  100
elif [ "$1" = "14" ]; then
    python src/bilevin/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  problems/pancake14/ \
    --seed 658 \
    --width  14 \
    --n-problems-per-stage  5000 \
    --randomize-curriculum-steps \
    --stages-multiple  4 \
    --n-problems-final-stage 50000 \
    --n-stages  8 \
    --n-valid  1000 \
    --n-test 1000 \
    --randomize-test-steps \
    --test-steps  500
elif [ "$1" = "18" ]; then
    python src/bilevin/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  problems/pancake18/ \
    --seed 358 \
    --width  18 \
    --n-problems-per-stage  5000 \
    --randomize-curriculum-steps \
    --stages-multiple  4 \
    --n-problems-final-stage 50000 \
    --n-stages  10 \
    --n-valid  1000 \
    --n-test 1000 \
    --randomize-test-steps \
    --test-steps  500
elif [ "$1" = "22" ]; then
    python src/bilevin/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  problems/pancake22/ \
    --seed 2321 \
    --width  22 \
    --n-problems-per-stage  1000 \
    --randomize-curriculum-steps \
    --stages-multiple  5 \
    --n-problems-final-stage 50000 \
    --n-stages  1 \
    --n-valid  1000 \
    --n-test 1000 \
    --randomize-test-steps \
    --test-steps  500
else
    usage
fi
