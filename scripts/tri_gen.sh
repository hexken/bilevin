#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR/../src/bilevin
cd $SCRIPT_DIR/../

usage() {
    echo "Usage: $0 [d|4|5|6]"
    exit 1
}

if [ $# -ne 1 ]; then
    usage
fi

if [ "$1" = "d" ]; then
    python src/bilevin/domains/wit_puzzle_generator.py \
    --puzzle  "triangles" \
    --seed  1743 \
    --width  4 \
    --output-path  problems2/tri4d/ \
    --n-stages 1 \
    --n-problems-per-stage 1 \
    --marker-prob-limits  0.1 0.5 \
    --min-path-ratio-limits 0.0 2.0 \
    --n-problems-final-stage 250 \
    --test-marker-prob 0.6 \
    --test-min-path-ratio 2.0 \
    --n-valid  100 \
    --n-test  100
elif [ "$1" = "4" ]; then
    python src/bilevin/domains/wit_puzzle_generator.py \
    --puzzle  "triangles" \
    --seed  1743 \
    --width  4 \
    --output-path  problems/tri4/ \
    --n-stages 1 \
    --n-problems-per-stage 1 \
    --marker-prob-limits  0.1 0.5 \
    --min-path-ratio-limits 0.0 2.0 \
    --n-problems-final-stage 50000 \
    --test-marker-prob 0.6 \
    --test-min-path-ratio 2.0 \
    --n-valid  1000 \
    --n-test  1000
elif [ "$1" = "5" ]; then
    python src/bilevin/domains/wit_puzzle_generator.py \
    --puzzle  "triangles" \
    --seed  3743 \
    --width  5 \
    --output-path  problems/tri5/ \
    --n-stages 1 \
    --n-problems-per-stage 1 \
    --marker-prob-limits  0.1 0.5 \
    --min-path-ratio-limits 0.0 2.0 \
    --n-problems-final-stage 50000 \
    --test-marker-prob 0.6 \
    --test-min-path-ratio 2.0 \
    --n-valid  1000 \
    --n-test  1000
else
    usage
fi
