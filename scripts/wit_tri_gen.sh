#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR/../src
cd $SCRIPT_DIR/../

if [ $# -ne 1 ]; then
    echo "Usage: $0 [d|4|5|6]"
    exit 1
fi

if [ "$1" = "d" ]; then
    python src/domains/wit_puzzle_generator.py \
    --puzzle  "triangles" \
    --seed  1643 \
    --width  4 \
    --output-path  problems/wit_tri4_debug/ \
    --marker-prob  0.6 \
    --min-path-ratio 0.8 \
    --n-train 500 \
    --n-valid  100 \
    --n-test  100
elif [ "$1" = "4" ]; then
    python src/domains/wit_puzzle_generator.py \
    --puzzle  "triangles" \
    --seed  1743 \
    --width  4 \
    --output-path  problems/wit_tri4/ \
    --marker-prob  0.6 \
    --min-path-ratio 0.8 \
    --n-train 50000 \
    --n-valid  1000 \
    --n-test  1000
elif [ $1 = "5" ]; then
    python src/domains/wit_puzzle_generator.py \
    --puzzle  "triangles" \
    --seed  1657 \
    --width  5 \
    --output-path  problems/wit_tri5/ \
    --marker-prob  0.55 \
    --min-path-ratio 0.75 \
    --n-train 50000 \
    --n-valid  1000 \
    --n-test  1000
elif [ $1 = "6" ]; then
    python src/domains/wit_puzzle_generator.py \
    --puzzle  "triangles" \
    --seed  936 \
    --width  6 \
    --output-path  problems/wit_tri6/ \
    --marker-prob  0.5 \
    --min-path-ratio 0.7 \
    --n-train 50 \
    --n-valid  1000 \
    --n-test  1000
fi


