#!/bin/bash
#
export OMP_NUM_THREADS=1

python src/main.py \
    --world-size 4 \
    --mode "test" \
    --agent Levin \
    --problemset-path problems/stp/4w-cur/16-test.json \
    --initial-budget 2000 \
    --seed 1 \
    --wandb-mode disabled \
    --model-path stp_m19/SlidingTilePuzzle-4w-cur2-32000-train_Levin-16000_1_1679259806 \
    # --update-levin-costs \
