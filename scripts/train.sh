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
    --curriculum-epochs 1 \
    --permutation-epochs 5 \
    --permutation-focus \
    --epoch-reduce-lr 99 \
    --epoch-reduce-grad-steps 2 \
    --epoch-begin-validate 1 \
    --expansion-budget 500 \
    --time-budget 10 \
    --grad-steps 10 \
    --batch-size-train 4 \
    --seed 1 \
    --wandb-mode online \
    --problemset-path problems/stp/april/3w/5000-train.json \
    --validset-path problems/stp/april/3w/1000-valid.json \
    # --problemset-path problems/stp/april/5w/50000-train.json \
    # --validset-path problems/stp/april/5w/4000-valid.json \

    # --problemset-path problems/witness/april/7w4c/50000-train.json \
    # --validset-path problems/witness/april/7w4c/4000-valid.json \
    # --model-path runs/SlidingTilePuzzle-3w-5000-train_Levin-500_1_1681246055 \
    #  --model-suffix  "best" \
     # --model-path runs/SlidingTilePuzzle-3w-20000-train_BiLevin-1000_1_1681165195 \
    # --problemset-path problems/witness/7w4c/5000-train.json \
    # --validset-path problems/witness/7w4c/500-valid.json \
    # --problemset-path problems/sokoban/unfiltered/trainsmall/train.json \
    # --validset-path problems/sokoban/unfiltered/validsmall/valid.json \
    # --include-prev-difficulty \
    # --exp-name "_orig" \
    # --update-levin-costs \
