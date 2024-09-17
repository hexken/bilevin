#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR/../src/bilevin
cd $SCRIPT_DIR/../

usage() {
    echo "Usage: $0 [d|14|18]"
    exit 1
}
if [ $# -ne 1 ]; then
    usage
fi

if [ "$1" = "d" ]; then
    python src/bilevin/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  problems/pancaked/ \
    --seed 4238 \
    --width  10 \
    --random-goal \
    --randomize-test-steps \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 100 \
    --n-valid  10 \
    --n-test 10
elif [ "$1" = "10rg" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  problems/pancake10rg/ \
    --seed 4238 \
    --width  10 \
    --random-goal \
    --randomize-test-steps \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 500 \
    --n-valid  100 \
    --n-test 100
elif [ "$1" = "12rg" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  problems/pancake12rg/ \
    --seed 3316 \
    --width  12 \
    --random-goal \
    --randomize-test-steps \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 500 \
    --n-valid  100 \
    --n-test 100
elif [ "$1" = "10" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  problems/pancake10/ \
    --seed 9298 \
    --width  10 \
    --permutation-test \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 500 \
    --n-valid  100 \
    --n-test 100
elif [ "$1" = "12" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  problems/pancake12/ \
    --seed 1016 \
    --width  12 \
    --permutation-test \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 500 \
    --n-valid  100 \
    --n-test 100
else
    usage
fi
