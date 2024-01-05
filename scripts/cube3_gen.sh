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
    --output-path  problems/cube3d/ \
    --n-problems-per-stage  10 \
    --stages-multiple  1 \
    --n-stages  100 \
    --randomize-curriculum-steps \
    --n-problems-final-stage  250 \
    --n-valid  100 \
    --n-test  100 \
    --test-steps 100 \
    --randomize-test-steps
elif [ "$1" = "3" ]; then
    python src/domains/puzzle_generator.py \
    --domain cube3 \
    --seed 3541 \
    --output-path  problems/cube3/ \
    --n-problems-per-stage  500 \
    --stages-multiple  1 \
    --n-stages  51 \
    --n-problems-final-stage  25000 \
    --n-valid  1000 \
    --n-test  1000 \
    --test-steps  50
    --randomize-test-steps
else
    usage
fi
