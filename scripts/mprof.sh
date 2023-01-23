#!/bin/bash

export OMP_NUM_THREADS=1

mprof run \
    src/main.py \
    --mode train \
    --agent BiLevin \
    --loss levin_loss_sum \
    --problemset-path problems/stp/3w/50000.json \
    --initial-budget 3000 \
    --grad-steps 10 \
    --batch-size-bootstrap 4 \
    --seed 1 \
    --wandb-mode disabled \
