#!/bin/bash
#
export OMP_NUM_THREADS=1

python bilevin/main.py \
    --agent PHS \
    --world-size 4 \
    --mode "test" \
    --test-path problems/stp4/50-test.pkl \
    --test-expansion-budget 7000 \
    --increase-budget \
    --model-path runs/stp4-100-train_PHS_1_1719983671/model_best_expanded.pt\
    --exp-name ""
