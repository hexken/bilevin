#!/bin/bash

scalene \
    src/main.py \
    --mode train \
    --agent BiLevin \
    --loss levin_loss_sum \
    --model-path trained_models/ \
    --domain Witness \
    --problems-path problems/witness/4x4_4_1000/ \
    --initial-budget 2000 \
    --grad-steps 10 \
    --batch-size-bootstrap 4 \
    --seed 1 \
    --track-params \
    --wandb-mode disabled

