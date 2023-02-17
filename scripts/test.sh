#!/bin/bash
#
export OMP_NUM_THREADS=1

python src/main.py \
    --world-size 4 \
    --mode "test" \
    --agent BiLevin \
    --problemset-path problems/witness/4w4c/50000-original.json \
    --initial-budget 2000 \
    --grad-steps 10 \
    --seed 1 \
    --wandb-mode disabled \
    # --model-path bugfixed2/Witness-4w4c-50000-original_Levin-2000_1_1675411815 \
    # --update-levin-costs \
