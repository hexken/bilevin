#!/bin/bash

export OMP_NUM_THREADS=1

python src/main.py \
    --world-size 4 \
    --mode train \
    --agent BiLevin \
    --loss-fn cross_entropy_avg_loss\
    --feature-net-lr 0.001 \
    --forward-feature-net-lr 0.001 \
    --backward-feature-net-lr 0.001 \
    --forward-policy-lr 0.001 \
    --backward-policy-lr 0.001 \
    --batch-begin-validate 1 \
    --validate-every 1000 \
    --checkpoint-every 10 \
    --train-expansion-budget 2500 \
    --test-expansion-budget 2500 \
    --time-budget 300 \
    --grad-steps 10 \
    --seed 1 \
    --runsdir-path runs/ \
    --max-expansion-budget 2500 \
    --min-samples-per-stage 10000 \
    --min-solve-ratio-stage 0 \
    --min-solve-ratio-exp 0 \
    --increase-budget \
    --n-tail 1000 \
    --problems-path problems/stp3/10000-train.pkl \
    --valid-path problems/stp3/100-valid.pkl \
    # --checkpoint-path runs/cube3-150-train_BiLevin_e2100_t300.0_1_1698874122/checkpoint_b15.pkl \
