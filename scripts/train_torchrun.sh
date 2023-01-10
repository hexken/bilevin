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
    --model-path trained_models/ \
    --problemset-path problems/witness/4w4c/1000.json \
    --initial-budget 2000 \
    --grad-steps 10 \
    --batch-size-bootstrap 4 \
    --seed 1 \
    --track-params \
    --wandb-mode disabled
