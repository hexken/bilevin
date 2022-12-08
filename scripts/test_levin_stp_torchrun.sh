#!/bin/bash

torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --master_addr=127.0.0.1 \
    --master_port=1234 \
    src/main.py \
    --mode train \
    --agent Levin \
    --loss cross_entropy_loss \
    --model-path trained_models/ \
    --domain SlidingTile \
    --problems-path problems/stp_test/3x3_20/ \
    --initial-budget 7000 \
    --grad-steps 10 \
    --batch-size-bootstrap 4 \
