#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR/../
cd $SCRIPT_DIR/../../

python src/domains/puzzle_generator.py \
--domain stp \
--seed \
4325 \
--width \
4 \
--output-path \
problems/stp4/ \
--n-problems-per-stage \
50000 \
--randomize-curriculum-steps \
--stages-multiple \
1000 \
--num-stages \
1 \
--n-valid \
1000 \
--n-test \
1000 \
--test-steps \
1000 \

python src/domains/puzzle_generator.py \
--domain stp \
--seed \
7546 \
--width \
5 \
--output-path \
problems/stp5/ \
--n-problems-per-stage \
50000 \
--randomize-curriculum-steps \
--stages-multiple \
1000 \
--num-stages \
1 \
--n-valid \
1000 \
--n-test \
1000 \
--test-steps \
1000 \
