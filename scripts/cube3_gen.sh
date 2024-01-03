#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR/../src
cd $SCRIPT_DIR/../

usage() {
    echo "Usage: $0 [d|3]"
    exit 1
}
if [ $# -gt 1 ]; then
    usage
fi

if [ "$1" = "d" ]; then
    python src/domains/puzzle_generator.py \
    --domain cube3 \
    --seed 3541 \
    --output-path  problems/cube3_debug/ \
    --n-problems-per-stage  200 \
    --randomize-curriculum-steps \
    --stages-multiple  5 \
    --num-stages  10 \
    --final-stage \
    --n-valid  100 \
    --n-test  100 \
    --test-steps 100
else
    python src/domains/puzzle_generator.py \
    --domain cube3 \
    --seed 3541 \
    --output-path  problems/cube3/ \
    --n-problems-per-stage  100000 \
    --randomize-curriculum-steps \
    --stages-multiple  5 \
    --num-stages  10 \
    --final-stage \
    --n-valid  1000 \
    --n-test  1000 \
    --test-steps  100
else
    usage
fi
