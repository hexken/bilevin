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
    python bilevin/domains/wit_puzzle_generator.py \
    --puzzle  "triangles" \
    --seed  2723 \
    --width  4 \
    --output-path  new_problems/trid/ \
    --random-start \
    --marker-prob 0.7 \
    --n-train 100 \
    --n-valid  10 \
    --n-test  10
elif [ "$1" = "4" ]; then
    python bilevin/domains/wit_puzzle_generator.py \
    --puzzle  "triangles" \
    --seed  2723 \
    --width  4 \
    --output-path  new_problems/tri4/ \
    --random-start \
    --marker-prob 0.7 \
    --n-train 50000 \
    --n-valid  1000 \
    --n-test  1000
elif [ "$1" = "5" ]; then
    python bilevin/domains/wit_puzzle_generator.py \
    --puzzle  "triangles" \
    --seed  6413 \
    --width  5 \
    --output-path  new_problems/tri5/ \
    --random-start \
    --marker-prob 0.7 \
    --n-train 50000 \
    --n-valid  1000 \
    --n-test  1000
else
    usage
fi
