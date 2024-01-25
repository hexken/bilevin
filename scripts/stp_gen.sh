#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR/../src/bilevin
cd $SCRIPT_DIR/../

usage() {
    echo "Usage: $0 [d|4|5|6]"
    exit 1
}
if [ $# -ne 1 ]; then
    usage
fi

if [ "$1" = "d" ]; then
    python src/bilevin/domains/puzzle_generator.py \
    --domain stp \
    --output-path  problems/stp4d/ \
    --seed 658 \
    --width  4 \
    --n-problems-per-stage  100 \
    --randomize-curriculum-steps \
    --stages-multiple  20 \
    --n-problems-final-stage 2500 \
    --n-stages  26 \
    --n-valid  100 \
    --n-test 100 \
    --randomize-test-steps \
    --test-steps  500
elif [ "$1" = "4" ]; then
    python src/bilevin/domains/puzzle_generator.py \
    --domain stp \
    --output-path  problems/stp4/ \
    --seed 658 \
    --width  4 \
    --n-problems-per-stage  1000 \
    --randomize-curriculum-steps \
    --stages-multiple  20 \
    --n-problems-final-stage 25000 \
    --n-stages  26 \
    --n-valid  1000 \
    --n-test 1000 \
    --randomize-test-steps \
    --test-steps  500
elif [ "$1" = "5" ]; then
    python src/bilevin/domains/puzzle_generator.py \
    --domain stp \
    --output-path  problems/stp5/ \
    --seed 126 \
    --width  5 \
    --n-problems-per-stage  1000 \
    --randomize-curriculum-steps \
    --stages-multiple  20 \
    --n-problems-final-stage 25000 \
    --n-stages  26 \
    --n-valid  1000 \
    --n-test 1000 \
    --randomize-test-steps \
    --test-steps  500
else
    usage
fi
