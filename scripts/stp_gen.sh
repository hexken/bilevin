#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR/../bilevin
cd $SCRIPT_DIR/../

usage() {
    echo "Usage: $0 [d|4|5|6]"
    exit 1
}
if [ $# -ne 1 ]; then
    usage
fi

if [ "$1" = "d" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain stp \
    --output-path  problems/stp5d/ \
    --seed 658 \
    --width  5 \
    --n-problems-per-stage  100 \
    --randomize-curriculum-steps \
    --stages-multiple  20 \
    --n-problems-final-stage 1000 \
    --n-stages  1 \
    --n-valid  100 \
    --n-test 100 \
    --randomize-test-steps \
    --test-steps  1000
elif [ "$1" = "4" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain stp \
    --output-path  problems/lelis/stp4/ \
    --seed 4238 \
    --width  4 \
    --randomize-curriculum-steps \
    --n-problems-per-stage  50000 \
    --stages 50 1000 \
    --stages-multiple 0 \
    --n-problems-final-stage 0 \
    --n-stages  0 \
    --n-valid  1000 \
    --n-test 1000 \
    --test-permutation
elif [ "$1" = "5" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain stp \
    --output-path  problems/stp5/ \
    --seed 1274 \
    --width  5 \
    --n-problems-per-stage  5000 \
    --randomize-curriculum-steps \
    --stages-multiple  10 \
    --n-problems-final-stage 50000 \
    --n-stages  11 \
    --n-valid  1000 \
    --n-test 1000 \
    --randomize-test-steps \
    --test-steps  500
elif [ "$1" = "5v" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain stp \
    --output-path  problems/lelis/stp5 \
    --exclude-path  problems/lelis/stp5/50000-train.pkl \
    --exclude-path  problems/lelis/stp5/1000-test.pkl \
    --seed 1274 \
    --width  5 \
    --n-problems-per-stage  0 \
    --randomize-curriculum-steps \
    --stages-multiple  10 \
    --n-problems-final-stage 0 \
    --n-stages  0 \
    --n-valid  1000 \
    --n-test 0 \
    --test-permutation
else
    usage
fi
