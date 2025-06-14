#!/bin/bash

export OMP_NUM_THREADS=1

flr=0.0001
blr=0.0001
python bilevin/main.py \
    --n-epochs 10 \
    --agent BiAStarBFS \
    --loss-fn default \
    --runsdir-path runs/ \
    --shuffle \
    --master-port 34568 \
    --seed 1 \
    --world-size 4 \
    --batch-size 8 \
    --n-eval 32 \
    --feature-net-type conv \
    --no-feature-net f \
    --share-feature-net f \
    --mode train \
    --weight-astar 1.0 \
    --optimizer Adam \
    --grad-steps 10 \
    \
    --n-kernels 32 \
    --kernel-size 1 2 \
    \
    --forward-feature-net-lr $flr \
    --forward-policy-layers 128 \
    --forward-policy-lr $flr \
    --forward-heuristic-layers 128 \
    --forward-heuristic-lr $flr \
    \
    --backward-feature-net-lr $blr \
    --backward-policy-layers 256 198 128 \
    --backward-policy-lr $blr \
    --backward-heuristic-layers 256 198 128 \
    --backward-heuristic-lr $blr \
    \
    \
    --time-budget 300 \
    --train-expansion-budget 4000 \
    \
    --train-path problems/stp4/50000-train.pkl \
    --valid-path problems/stp4/1000-valid.pkl \
    --test-path problems/stp4/1000-test.pkl \
    # --checkpoint-path ~/Envs/re/checkpoint_e8b800.pkl \
    # --mask-invalid-actions \
    # --train-path problems/stp4/50000-train.pkl \
    # --valid-path problems/stp4/1000-valid.pkl \
    # --test-path problems/stp4/1000-test.pkl \
    # --checkpoint-every-n-batch 5 \
    #
