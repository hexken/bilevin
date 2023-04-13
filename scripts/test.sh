#!/bin/bash
#
export OMP_NUM_THREADS=1

python src/main.py \
    --world-size 4 \
    --mode "test" \
    --agent BiLevin \
    --problemset-path problems/stp/april/3w/1000-test.json \
    --initial-budget 500 \
    --seed 1 \
    --wandb-mode disabled \
    --model-path runs/SlidingTilePuzzle-3w-20000-train_BiLevin-1000_1_1681165195 \
    --model-suffix  "best" \
    # --update-levin-costs \
