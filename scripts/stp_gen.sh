#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR/../src
cd $SCRIPT_DIR/../

if [ $# -ne 1 ]; then
    echo "Usage: $0 [d|4|5]"
    exit 1
fi

if [ "$1" = "d" ]; then
    python src/domains/puzzle_generator.py \
    --domain stp \
    --output-path  problems/stp4_debug/ \
    --seed 658 \
    --width  4 \
    --n-problems-per-stage  500 \
    --randomize-curriculum-steps \
    --stages-multiple  100 \
    --num-stages  1 \
    --n-valid  100 \
    --n-test 100 \
    --randomize-test-steps \
    --test-steps  1000
elif [ "$1" = "4" ]; then
    python src/domains/puzzle_generator.py \
    --domain stp \
    --output-path  problems/stp4/ \
    --seed 658 \
    --width  4 \
    --n-problems-per-stage  50000 \
    --randomize-curriculum-steps \
    --stages-multiple  1000 \
    --num-stages  1 \
    --n-valid  1000 \
    --n-test 1000 \
    --randomize-test-steps \
    --test-steps  1000
elif [ "$1" = "4c" ]; then
    python src/domains/puzzle_generator.py \
    --domain stp \
    --output-path  problems/stp4c/ \
    --seed 658 \
    --width  4 \
    --n-problems-per-stage  5000 \
    --randomize-curriculum-steps \
    --stages-multiple  100 \
    --num-stages  10 \
    --n-valid  1000 \
    --n-test 1000 \
    --randomize-test-steps \
    --test-steps  1000
elif [ "$1" = "4cd" ]; then
    python src/domains/puzzle_generator.py \
    --domain stp \
    --output-path  problems/stp4c_debug/ \
    --seed 658 \
    --width  4 \
    --n-problems-per-stage  100 \
    --randomize-curriculum-steps \
    --stages-multiple  10 \
    --num-stages  10 \
    --n-valid  100 \
    --n-test 100 \
    --randomize-test-steps \
    --test-steps  100
elif [ $1 = "5" ]; then
    python src/domains/puzzle_generator.py \
    --domain stp \
    --output-path  problems/stp5/ \
    --seed 338 \
    --width  5 \
    --n-problems-per-stage  50000 \
    --randomize-curriculum-steps \
    --stages-multiple  1000 \
    --num-stages  1 \
    --n-valid  1000 \
    --n-test 1000 \
    --randomize-test-steps \
    --test-steps  1000
fi
