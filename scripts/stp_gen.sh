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
    --output-path  problems/stpd/ \
    --seed 4238 \
    --width  4 \
    --random-goal \
    --randomize-test-steps \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 100 \
    --n-valid  10 \
    --n-test 10
elif [ "$1" = "4rg" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain stp \
    --output-path  new_problems/stp4rg/ \
    --seed 4238 \
    --width  4 \
    --random-goal \
    --randomize-test-steps \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 500 \
    --n-valid  100 \
    --n-test 100
elif [ "$1" = "5rg" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain stp \
    --output-path  new_problems/stp5rg/ \
    --seed 1298 \
    --width  4 \
    --random-goal \
    --randomize-test-steps \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 500 \
    --n-valid  100 \
    --n-test 100
elif [ "$1" = "4" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain stp \
    --output-path  new_problems/stp4/ \
    --seed 5238 \
    --width  4 \
    --permutation-test \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 500 \
    --n-valid  100 \
    --n-test 100
elif [ "$1" = "5" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain stp \
    --output-path  new_problems/stp5/ \
    --seed 7228 \
    --width  4 \
    --permutation-test \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 500 \
    --n-valid  100 \
    --n-test 100
else
    usage
fi
