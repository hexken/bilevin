#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR/../src
cd $SCRIPT_DIR/../

python src/domains/puzzle_generator.py \
--domain stp \
--output-path  problems/stp4_debug/ \
--seed 658 \
--width  4 \
--n-problems-per-stage  500 \
--randomize-curriculum-steps \
--stages-multiple  100 \
--num-stages  1 \
--n-valid  100 \
--n-test 100 \
--randomize-test-steps \
--test-steps  1000 \

python src/domains/puzzle_generator.py \
--domain stp \
--output-path  problems/stp4/ \
--seed 658 \
--width  4 \
--n-problems-per-stage  50000 \
--randomize-curriculum-steps \
--stages-multiple  10 \
--num-stages  1 \
--n-valid  1000 \
--n-test 1000 \
--randomize-test-steps \
--test-steps  1000 \

python src/domains/puzzle_generator.py \
--domain stp \
--output-path  problems/stp5/ \
--seed 338 \
--width  4 \
--n-problems-per-stage  50000 \
--randomize-curriculum-steps \
--stages-multiple  10 \
--num-stages  1 \
--n-valid  1000 \
--n-test 1000 \
--randomize-test-steps \
--test-steps  1000