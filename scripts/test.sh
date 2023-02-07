#!/bin/bash
#
export OMP_NUM_THREADS=1

python src/main.py \
    --world-size 2 \
    --mode "test" \
    --agent Levin \
    --loss levin_loss_sum \
    --problemset-path problems/witness/4w4c/debug.json \
    --initial-budget 2000 \
    --grad-steps 10 \
    --batch-size-print 4 \
    --seed 1 \
    --wandb-mode online \
    --model-path bugfixed2/Witness-4w4c-50000-original_Levin-2000_1_1675411815 \
    # --update-levin-costs \
