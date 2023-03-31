#!/bin/bash
#
export OMP_NUM_THREADS=1

python src/main.py \
    --world-size 4 \
    --mode train \
    --agent BiLevin \
    --loss cross_entropy_loss \
    --feature-net-lr 0.001 \
    --forward-policy-lr 0.001 \
    --backward-policy-lr 0.001 \
    --bootstrap-epochs 0 \
    --curriculum-epochs 2 \
    --permutation-epochs 0 \
    --epochs-reduce-lr 4 \
    --epoch-begin-validate 1 \
    --initial-budget 16000 \
    --grad-steps 10 \
    --batch-size-train 4 \
    --seed 1 \
    --wandb-mode disabled \
    --permutation-focus \
    --problemset-path problems/witness/7w4c/5000-train.json \
    --validset-path problems/witness/7w4c/500-valid.json \
    # --problemset-path problems/sokoban/unfiltered/trainsmall/train.json \
    # --validset-path problems/sokoban/unfiltered/validsmall/valid.json \
     # --problemset-path problems/stp/4w-small/260-train.json \
     # --validset-path problems/stp/4w-small/100-valid.json \
    # --include-prev-difficulty \
    # --exp-name "_orig" \
    # --update-levin-costs \
