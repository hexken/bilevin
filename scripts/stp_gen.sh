#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR/../bilevin
cd $SCRIPT_DIR/../


if [ "$1" = "d" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain stp \
    --output-path  new_problems/stpd/ \
    --seed 4238 \
    --width  4 \
    --random-goal \
    --randomize-test-steps \
    --min-steps 50 \
    --max-steps 200 \
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
    --n-train 50000 \
    --n-valid  1000 \
    --n-test 1000
elif [ "$1" = "5rg" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain stp \
    --output-path  new_problems/stp5rg/ \
    --seed 1298 \
    --width  5 \
    --random-goal \
    --randomize-test-steps \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 50000 \
    --n-valid  1000 \
    --n-test 1000
elif [ "$1" = "4" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain stp \
    --output-path  new_problems/stp4/ \
    --seed 5238 \
    --width  4 \
    --permutation-test \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 50000 \
    --n-valid  1000 \
    --n-test 1000
elif [ "$1" = "5" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain stp \
    --output-path  new_problems/stp5/ \
    --seed 7228 \
    --width  5 \
    --permutation-test \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 50000 \
    --n-valid  1000 \
    --n-test 1000
else
    echo "Invalid argument"
fi
