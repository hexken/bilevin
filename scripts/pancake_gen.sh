#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR/../src
cd $SCRIPT_DIR/../

python src/domains/puzzle_generator.py \
--domain pancake \
--output-path  problems/pancake4_debug/ \
--seed 658 \
--width  6 \
--n-problems-per-stage  500 \
--randomize-curriculum-steps \
--stages-multiple  100 \
--num-stages  1 \
--n-valid  100 \
--n-test 100 \
--randomize-test-steps \
--test-steps  1000 \
