#!/bin/bash
#
export OMP_NUM_THREADS=1

python src/main.py \
    --world-size 4 \
    --mode train \
    --agent BiLevin \
    --loss levin_loss \
    --learning-rate 0.001 \
    --problemset-path problems/stp/4w-small/8-train.json \
    --validset-path problems/stp/4w-small/16-valid.json \
    --bootstrap-epochs 1 \
    --curriculum-epochs 1 \
    --permutation-epochs 1 \
    --epochs-reduce-lr 1 \
    --epoch-begin-validate 1 \
    --initial-budget 2 \
    --grad-steps 10 \
    --batch-size-train 4 \
    --seed 1 \
    --wandb-mode online \
    # --exp-name "_orig" \
    # --update-levin-costs \
