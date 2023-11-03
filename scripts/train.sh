#!/bin/bash

export OMP_NUM_THREADS=1

python src/main.py \
    --world-size 4 \
    --mode train \
    --agent BiLevin \
    --loss-fn cross_entropy_avg_loss\
    --cost-fn levin_cost \
    --feature-net-lr 0.001 \
    --forward-feature-net-lr 0.001 \
    --backward-feature-net-lr 0.001 \
    --forward-policy-lr 0.001 \
    --backward-policy-lr 0.001 \
    --batch-begin-validate 1 \
    --validate-every 10 \
    --checkpoint-every 5 \
    --train-expansion-budget 2 \
    --test-expansion-budget 500 \
    --time-budget 300 \
    --grad-steps 10 \
    --seed 1 \
    --runsdir-path runs/ \
    --min-samples-per-stage 100 \
    --min-solve-ratio 0 \
    --n-solve-ratio 0 \
    --problems-path problems/cube3/1100000-train.pkl \
    --valid-path problems/cube3/1000-valid.pkl \
    # --checkpoint-path runs/cube3-150-train_BiLevin_e2100_t300.0_1_1698874122/checkpoint_b15.pkl \
