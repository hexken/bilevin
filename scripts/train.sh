#!/bin/bash

export OMP_NUM_THREADS=1

python src/main.py \
    --runsdir-path runs/ \
    --problems-path problems/pancake4_debug/500-train.pkl \
    --valid-path problems/pancake4_debug/100-valid.pkl \
    --no-feature-net \
    --seed 1 \
    --world-size 1 \
    --mode train \
    --agent AStar \
    --loss-fn mse_loss \
    --grad-steps 10 \
    \
    --share-feature-net \
    --num-kernels 32 \
    --kernel-size 1 2 \
    \
    --conditional-backward \
    \
    --forward-feature-net-lr 0.001 \
    --forward-policy-layers 128 \
    --forward-policy-lr 0.001 \
    --forward-heuristic-layers 128 \
    --forward-heuristic-lr 0.001 \
    \
    --backward-feature-net-lr 0.001 \
    --backward-policy-layers 256 198 128 \
    --backward-policy-lr 0.001 \
    --backward-heuristic-layers 256 298 128 \
    --backward-heuristic-lr 0.001 \
    \
    --batch-begin-validate 1 \
    --validate-every 1000 \
    --checkpoint-every 10 \
    \
    --time-budget 300 \
    --train-expansion-budget 2500 \
    --max-expansion-budget 2500 \
    --test-expansion-budget 2500 \
    \
    --min-samples-per-stage 10000 \
    --min-solve-ratio-stage 0 \
    --min-solve-ratio-exp 0 \
    --increase-budget \
    \
    --n-tail 1000 \
    \
    # --checkpoint-path runs/cube3-150-train_BiLevin_e2100_t300.0_1_1698874122/checkpoint_b15.pkl \
