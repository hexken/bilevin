#!/bin/bash
#
export OMP_NUM_THREADS=1

python src/main.py \
    --world-size 4 \
    --mode "test" \
    --agent BiLevin \
    --problemset-path problems/stp/april/3w/1000-test.json \
    --expansion-budget 500 \
    --seed 1 \
    --wandb-mode disabled \
    --model-path runs/SlidingTilePuzzle-3w-5000-train_BiLevin-e500-t300_1_1682722368 \
    --model-suffix  "best_expanded" \
    # --update-levin-costs \
