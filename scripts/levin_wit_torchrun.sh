#!/bin/bash
#
export OMP_NUM_THREADS=1

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_addr=$(hostname)\
    --master_port=34567 \
    src/main.py \
    --mode train \
    --agent Levin \
    --loss levin_loss_sum \
    --model-path trained_models/ \
    --domain Witness \
    --problems-path problems/witness/puzzles_4x4_50k_train/ \
    --initial-budget 2000 \
    --grad-steps 10 \
    --batch-size-bootstrap 4 \
    # --track-params \
    # --wandb
