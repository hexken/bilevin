#!/bin/bash

torchrun \
    --nnodes=1 \
    --nproc_per_node=48 \
    --master_addr=$(hostname)\
    --master_port=34567 \
    src/main.py \
    --mode train \
    --agent Levin \
    --loss cross_entropy_loss \
    --model-path trained_models/ \
    --domain SlidingTile \
    --problems-path problems/stp_test/3x3_1000/ \
    --initial-budget 7000 \
    --grad-steps 10 \
    --batch-size-bootstrap 48 \
    --wandb
    # --cuda
