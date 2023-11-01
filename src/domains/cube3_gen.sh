#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR/../
cd $SCRIPT_DIR/../../
python src/domains/puzzle_generator.py \
--domain cube3 \
--seed \
9821 \
--output-path \
problems/cube3/ \
--n-problems-per-stage \
50 \
--randomize-curriculum-steps \
--stages-multiple \
5 \
--num-stages \
3 \
--n-valid \
100 \
--n-test \
100 \
--test-steps \
10 \
