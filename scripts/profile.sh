#!/bin/bash

scalene --outfile profile_witness.txt --cpu --memory \
    src/main.py \
    --- \
    --mode train \
    --agent Levin \
    --loss levin_loss_sum \
    --model-path trained_models/ \
    --problemset-path problems/stp/3w/50000.json \
    --initial-budget 5000 \
    --grad-steps 10 \
    --batch-size-bootstrap 4 \
    --seed 1 \
    --wandb-mode disabled

