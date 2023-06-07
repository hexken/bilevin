#!/bin/bash
#
export OMP_NUM_THREADS=1

python src/main.py \
    --world-size 4 \
    --mode train \
    --agent BiLevin \
    --loss levin_loss \
    --feature-net-lr 0.001 \
    --forward-policy-lr 0.001 \
    --backward-policy-lr 0.001 \
    --bootstrap-epochs 0 \
    --curriculum-epochs 2 \
    --permutation-epochs 0 \
    --epoch-reduce-lr 99 \
    --epoch-reduce-grad-steps 99 \
    --epoch-begin-validate 1 \
    --expansion-budget 8000 \
    --time-budget 300 \
    --grad-steps 10 \
    --batch-size-train 4 \
    --seed 1 \
    --wandb-mode disabled \
    --problemset-path problems/sokoban/unfiltered/april/train.json \
    --validset-path problems/sokoban/unfiltered/april/valid.json \
    # --problemset-path problems/witness/may/6w4c/50000-train.json \
    # --validset-path problems/witness/may/6w4c/4000-valid.json \
    # --problemset-path problems/stp/4w-debug/2000-train.json \
    # --validset-path problems/stp/4w-debug/1000-valid.json \
    # --problemset-path problems/stp/4w-simplecur/50000-train.json \
    # --validset-path problems/stp/4w-simplecur/1000-valid.json \
    # --problemset-path problems/witness/may/4w4c/50000-train.json \
    # --validset-path problems/witness/may/4w4c/4000-valid.json \
    # --problemset-path problems/stp/april/3w/5000-train.json \
    # --validset-path problems/stp/april/3w/1000-valid.json \
    # --problemset-path problems/stp/april/5w/50000-train.json \
    # --validset-path problems/stp/april/5w/4000-valid.json \

    # --model-path runs/SlidingTilePuzzle-3w-5000-train_Levin-500_1_1681246055 \
    #  --model-suffix  "best" \
     # --model-path runs/SlidingTilePuzzle-3w-20000-train_BiLevin-1000_1_1681165195 \
    # --problemset-path problems/witness/7w4c/5000-train.json \
    # --validset-path problems/witness/7w4c/500-valid.json \
    # --include-prev-difficulty \
    # --exp-name "_orig" \
    # --update-levin-costs \
