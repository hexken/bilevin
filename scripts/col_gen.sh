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
elif [ "$1" = "4" ]; then
    python bilevin/domains/wit_puzzle_generator.py \
    --puzzle  "colors" \
    --seed  2723 \
    --width  4 \
    --output-path  new_problems/cold/ \
    --random-start \
    --marker-prob 0.7 \
    --n-train 500 \
    --n-valid  100 \
    --n-test  100
elif [ "$1" = "4" ]; then
    python bilevin/domains/wit_puzzle_generator.py \
    --puzzle  "colors" \
    --seed  2723 \
    --width  4 \
    --output-path  new_problems/col4/ \
    --random-start \
    --marker-prob 0.7 \
    --n-train 500 \
    --n-valid  100 \
    --n-test  100
elif [ "$1" = "5" ]; then
    python bilevin/domains/wit_puzzle_generator.py \
    --puzzle  "colors" \
    --seed  2928 \
    --width  4 \
    --output-path  new_problems/col5/ \
    --random-start \
    --marker-prob 0.7 \
    --n-train 500 \
    --n-valid  100 \
    --n-test  100
else
    usage
fi
