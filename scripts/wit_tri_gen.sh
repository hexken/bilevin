#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR/../src
cd $SCRIPT_DIR/../

python src/domains/wit_puzzle_generator.py \
--puzzle  "triangles" \
--seed  43276 \
--width  4 \
--output-path  problems/wit_tri4/ \
--marker-prob  0.2 \
--min-path-ratio 0.35 \
--n-train 500 \
--n-valid  100 \
--n-test  100 \

# python src/domains/puzzle_generator.py \
# --domain stp \
# --seed \
# 4325 \
# --width \
# 4 \
# --output-path \
# problems/stp4/ \
# --n-problems-per-stage \
# 50000 \
# --randomize-curriculum-steps \
# --stages-multiple \
# 1000 \
# --num-stages \
# 1 \
# --n-valid \
# 1000 \
# --n-test \
# 1000 \
# --test-steps \
# 1000 \

# python src/domains/puzzle_generator.py \
# --domain stp \
# --seed \
# 7546 \
# --width \
# 5 \
# --output-path \
# problems/stp5/ \
# --n-problems-per-stage \
# 50000 \
# --randomize-curriculum-steps \
# --stages-multiple \
# 1000 \
# --num-stages \
# 1 \
# --n-valid \
# 1000 \
# --n-test \
# 1000 \
# --test-steps \
# 1000 \
