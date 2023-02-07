#!/bin/bash
#
export OMP_NUM_THREADS=1

python src/main.py \
--world-size 4 \
--mode train \
--agent BiLevin \
--loss levin_loss_sum \
--problemset-path problems/witness/4w4c/50000-original.json \
--initial-budget 2000 \
--grad-steps 10 \
--batch-size-bootstrap 4 \
--seed 1 \
--wandb-mode disabled \
# --exp-name "_orig" \
# --update-levin-costs \
