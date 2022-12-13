#!/bin/bash
#
export OMP_NUM_THREADS=1

torchrun \
    --nnodes=1 \
    --nproc_per_node=48 \
    --master_addr=$(hostname)\
    --master_port=34567 \
    src/main.py \
    --mode train \
    --agent Levin \
    --loss levin_loss_sum \
    --model-path trained_models/ \
    --domain SlidingTile \
    --problems-path problems/stp/puzzles_5x5_train/ \
    --initial-budget 7000 \
    --grad-steps 10 \
    --batch-size-bootstrap 48 \
    --wandb
    # --cuda
