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
    --mode "test" \
    --agent Levin \
    --loss levin_loss_sum \
    --problemset-path problems/witness/4w4c/debug.json \
    --initial-budget 2000 \
    --grad-steps 10 \
    --batch-size-bootstrap 4 \
    --seed 1 \
    --wandb-mode disabled \
    --model-path wit_4w4c_tb/Witness-4w4c-50000-original_Levin-2000_1_1674625065 \
    # --update-levin-costs \
