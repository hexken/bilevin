#!/bin/bash
#
export OMP_NUM_THREADS=1

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_addr=$(hostname)\
    --master_port=34567 \
    src/main.py \
    --mode train \
    --agent BiLevin \
    --loss levin_loss_sum \
    --problemset-path problems/stp/3w/50000.json \
    --initial-budget 2000 \
    --grad-steps 10 \
    --batch-size-bootstrap 4 \
    --seed 1 \
    --wandb-mode disabled \
    # --exp-name "_ULC" \
    # --update-levin-costs \
