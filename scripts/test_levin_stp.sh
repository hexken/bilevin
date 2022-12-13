#!/bin/bash

python src/main.py \
    --mode train \
    --agent Levin \
    --loss levin_loss \
    --model-path trained_models/ \
    --domain SlidingTile \
    --problems-path problems/stp_test/3x3_20/ \
    --initial-budget 7000 \
    --grad-steps 10 \
    --batch-size-bootstrap 4 \
    # --wandb \
