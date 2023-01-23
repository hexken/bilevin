#!/bin/bash

export OMP_NUM_THREADS=1

scalene --reduced-profile --html --outfile profile_stp_5w_levin.html --cpu --memory \
    --- \
    src/main.py \
    --mode train \
    --agent Levin \
    --loss levin_loss_sum \
    --problemset-path problems/stp/5w/50000-original.json \
    --initial-budget 10000 \
    --grad-steps 10 \
    --batch-size-bootstrap 4 \
    --seed 1 \
    --wandb-mode disabled \
