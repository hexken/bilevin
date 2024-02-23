#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR/../src/bilevin
cd $SCRIPT_DIR/../

usage() {
    echo "Usage: $0 [d|3]"
    exit 1
}
if [ $# -gt 1 ]; then
    usage
fi

if [ "$1" = "d" ]; then
    python src/bilevin/domains/puzzle_generator.py \
    --domain cube3 \
    --seed 3541 \
    --output-path  problems/cube3d/ \
    --n-problems-per-stage  250 \
    --stages-multiple  1 \
    --n-stages  11 \
    --randomize-curriculum-steps \
    --n-problems-final-stage  500 \
    --n-valid  100 \
    --n-test  100 \
    --test-steps 100 \
    --randomize-test-steps
elif [ "$1" = "3" ]; then
    python src/bilevin/domains/puzzle_generator.py \
    --domain cube3 \
    --seed 3541 \
    --output-path  problems/cube3/ \
    --randomize-curriculum-steps \
    --n-problems-per-stage 25000 \
    --stages-multiple  2 \
    --n-stages 21 \
    --n-problems-final-stage 50000 \
    --n-valid  1000 \
    --n-test  1000 \
    --test-steps  50 \
    --randomize-test-steps
else
    usage
fi
