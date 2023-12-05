#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR/../src
cd $SCRIPT_DIR/../

python src/domains/wit_puzzle_generator.py \
--puzzle  "colors" \
--seed  12243 \
--width  4 \
--output-path  problems/wit_col4_debug/ \
--marker-prob  0.6 \
--min-path-ratio 0.8 \
--n-train 500 \
--n-valid  100 \
--n-test  100 \

python src/domains/wit_puzzle_generator.py \
--puzzle  "colors" \
--seed  143 \
--width  4 \
--output-path  problems/wit_col4/ \
--marker-prob  0.6 \
--min-path-ratio 0.8 \
--n-train 50000 \
--n-valid  1000 \
--n-test  1000 \

python src/domains/wit_puzzle_generator.py \
--puzzle  "colors" \
--seed  657 \
--width  5 \
--output-path  problems/wit_col5/ \
--marker-prob  0.6 \
--min-path-ratio 0.8 \
--n-train 50000 \
--n-valid  1000 \
--n-test  1000 \

python src/domains/wit_puzzle_generator.py \
--puzzle  "colors" \
--seed  876 \
--width  6 \
--output-path  problems/wit_col6/ \
--marker-prob  0.6 \
--min-path-ratio 0.8 \
--n-train 50000 \
--n-valid  1000 \
--n-test  1000 \


