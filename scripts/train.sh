#!/bin/bash
export OMP_NUM_THREADS=1

python src/main.py \
    --mode train \
    --agent Levin \
    --loss levin_loss_avg \
    --problemset-path problems/witness/4w4c/50000.json \
    --initial-budget 2000 \
    --grad-steps 10 \
    --batch-size-bootstrap 4 \
    --seed 1 \
    --wandb-mode disabled
