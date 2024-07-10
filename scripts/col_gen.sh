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
    --puzzle  "colors" \
    --seed  2723 \
    --width  4 \
    --output-path  problems/col4d/ \
    --n-stages 11 \
    --n-problems-per-stage 1000 \
    --marker-prob-limits  0.1 0.5 \
    --min-path-ratio-limits 0.0 2.0 \
    --n-problems-final-stage 2000 \
    --test-marker-prob 0.6 \
    --test-min-path-ratio 2.0 \
    --n-valid  100 \
    --n-test  100
elif [ "$1" = "4" ]; then
    python bilevin/domains/wit_puzzle_generator.py \
    --puzzle  "colors" \
    --seed  2723 \
    --width  4 \
    --output-path  problems/col4/ \
    --n-stages 1 \
    --n-problems-per-stage 0 \
    --marker-prob-limits  0.1 0.5 \
    --min-path-ratio-limits 0.0 2.0 \
    --n-problems-final-stage 50000 \
    --test-marker-prob 0.6 \
    --test-min-path-ratio 1.5 \
    --n-valid  1000 \
    --n-test  1000
elif [ "$1" = "5" ]; then
    python bilevin/domains/wit_puzzle_generator.py \
    --puzzle  "colors" \
    --seed  1753 \
    --width  5 \
    --output-path  problems/col5/ \
    --n-stages 1 \
    --n-problems-per-stage 0 \
    --marker-prob-limits  0.1 0.5 \
    --min-path-ratio-limits 0.0 2.0 \
    --n-problems-final-stage 50000 \
    --test-marker-prob 0.6 \
    --test-min-path-ratio 1.5 \
    --n-valid  1000 \
    --n-test  1000
else
    usage
fi
