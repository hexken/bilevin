#!/bin/bash
#
export OMP_NUM_THREADS=1

python src/main.py \
    --world-size 32 \
    --mode train \
    --agent BiLevin \
    --loss levin_loss \
    --learning-rate 0.001 \
    --problemset-path problems/witness/4w4c/250.json \
    --validset-path problems/witness/4w4c/100.json \
    --initial-budget 2000 \
    --grad-steps 10 \
    --batch-size-train 32 \
    --seed 1 \
    --wandb-mode offline \
    # --exp-name "_orig" \
    # --update-levin-costs \
