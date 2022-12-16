#!/bin/bash
python src/main.py \
    --mode train \
    --agent Levin \
    --loss levin_loss_sum \
    --model-path trained_models/ \
    --domain SlidingTile \
    --problems-path problems/stp/puzzles_5x5_train/ \
    --initial-budget 7000 \
    --grad-steps 10 \
    --batch-size-bootstrap 1 \
    # --cuda
    # --wandb \
