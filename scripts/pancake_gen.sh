#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR/../src
cd $SCRIPT_DIR/../

usage() {
    echo "Usage: $0 [d|8|12|16]"
    exit 1
}
if [ $# -ne 1 ]; then
    usage
fi

if [ "$1" = "d" ]; then
    python src/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  problems/pancake8_debug/ \
    --seed 658 \
    --width  8 \
    --n-problems-per-stage  500 \
    --randomize-curriculum-steps \
    --stages-multiple  100 \
    --num-stages  1 \
    --n-valid  100 \
    --n-test 100 \
    --randomize-test-steps \
    --test-steps  1000
elif [ "$1" = "8" ]; then
    python src/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  problems/pancake8 \
    --seed 328 \
    --width  8 \
    --n-problems-per-stage  50000 \
    --randomize-curriculum-steps \
    --stages-multiple  1000 \
    --num-stages  1 \
    --n-valid  1000 \
    --n-test 1000 \
    --randomize-test-steps \
    --test-steps  1000
elif [ "$1" = "12" ]; then
    python src/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  problems/pancake12 \
    --seed 128 \
    --width  12 \
    --n-problems-per-stage  50000 \
    --randomize-curriculum-steps \
    --stages-multiple  1000 \
    --num-stages  1 \
    --n-valid  1000 \
    --n-test 1000 \
    --randomize-test-steps \
    --test-steps  1000
elif [ "$1" = "16" ]; then
    python src/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  problems/pancake16 \
    --seed 518 \
    --width  16 \
    --n-problems-per-stage  50000 \
    --randomize-curriculum-steps \
    --stages-multiple  1000 \
    --num-stages  1 \
    --n-valid  1000 \
    --n-test 1000 \
    --randomize-test-steps \
    --test-steps  1000
else
    usage
fi
