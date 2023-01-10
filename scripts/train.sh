#!/bin/bash
# export OMP_NUM_THREADS=1
# apparently this runs faster WITHOUT letting openmp use multiple threads

python src/main.py \
    --mode train \
    --agent Levin \
    --loss levin_loss_avg \
    --model-path trained_models/ \
    --problems-path problems/witness/6x6_5/ \
    --initial-budget 2000 \
    --grad-steps 10 \
    --batch-size-bootstrap 1 \
    --seed 1 \
    --track-params \
    --wandb disabled
