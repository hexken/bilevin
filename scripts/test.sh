#!/bin/bash
#
export OMP_NUM_THREADS=1

python src/main.py \
    --world-size 4 \
    --mode "test" \
    --agent BiLevin \
    --problemset-path fresh_problems/stp/w4/10000-test.json \
    --expansion-budget 24000 \
    --seed 1 \
    --wandb-mode disabled \
    --model-path \
    --model-suffix  "best_expanded" \
    --exp-name ""
