#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR/../src
cd $SCRIPT_DIR/../

usage() {
    echo "Usage: $0 [d|4|5|6]"
    exit 1
}
if [ $# -ne 1 ]; then
    usage
fi

if [ "$1" = "d" ]; then
    python src/domains/wit_puzzle_generator.py \
    --puzzle  "colors" \
    --seed  1233 \
    --width  4 \
    --output-path  problems/col4d/ \
    --n-stages 26 \
    --n-problems-per-stage 100 \
    --marker-prob-limits  0.3 0.59 \
    --min-path-ratio-limits 0.4 0.79 \
    --n-problems-final-stage 2500 \
    --test-marker-prob 0.6 \
    --test-min-path-ratio 0.8 \
    --n-valid  100 \
    --n-test  100
elif [ "$1" = "4" ]; then
    python src/domains/wit_puzzle_generator.py \
    --puzzle  "colors" \
    --seed  2723 \
    --width  4 \
    --output-path  problems/col4/ \
    --n-stages 26 \
    --n-problems-per-stage 1000 \
    --marker-prob-limits  0.3 0.59 \
    --min-path-ratio-limits 0.4 0.79 \
    --n-problems-final-stage 25000 \
    --test-marker-prob 0.6 \
    --test-min-path-ratio 0.8 \
    --n-valid  1000 \
    --n-test  1000
elif [ $1 = "5" ]; then
    python src/domains/wit_puzzle_generator.py \
    --puzzle  "colors" \
    --seed  3652 \
    --width  5 \
    --output-path  problems/col5/ \
    --n-stages 26 \
    --n-problems-per-stage 1000 \
    --marker-prob-limits  0.3 0.59 \
    --min-path-ratio-limits 0.4 0.79 \
    --n-problems-final-stage 25000 \
    --test-marker-prob 0.55 \
    --test-min-path-ratio 0.75 \
    --n-valid  1000 \
    --n-test  1000
elif [ $1 = "6" ]; then
    python src/domains/wit_puzzle_generator.py \
    --puzzle  "colors" \
    --seed  9162 \
    --width  6 \
    --output-path  problems/col6/ \
    --n-stages 26 \
    --n-problems-per-stage 1000 \
    --marker-prob-limits  0.3 0.49 \
    --min-path-ratio-limits 0.3 0.69 \
    --test-marker-prob 0.5 \
    --test-min-path-ratio 0.7 \
    --n-problems-final-stage 25000 \
    --n-valid  1000 \
    --n-test  1000
else
    usage
fi
