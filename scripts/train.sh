#!/bin/bash
#
export OMP_NUM_THREADS=1
# export PYTORCH_JIT=0
# export PYTORCH_JIT_LOG_LEVEL=">>graph_fuser.cpp"

python src/main.py \
    --world-size 4 \
    --mode train \
    --agent BiLevin \
    --loss cross_entropy_loss \
    --feature-net-lr 0.001 \
    --forward-feature-net-lr 0.001 \
    --backward-feature-net-lr 0.001 \
    --forward-policy-lr 0.001 \
    --backward-policy-lr 0.001 \
    --bootstrap-epochs 0 \
    --curriculum-epochs 5 \
    --permutation-epochs 30 \
    --epoch-reduce-lr 99 \
    --epoch-reduce-grad-steps 99 \
    --epoch-begin-validate 1 \
    --expansion-budget 24000 \
    --time-budget 300 \
    --grad-steps 10 \
    --n-subgoals 5 \
    --batch-size-train 4 \
    --seed 1 \
    --wandb-mode disabled \
    --runsdir-path runs/ \
    --problemset-path fresh_problems/stp/w4/50000-train.json \
    --validset-path fresh_problems/stp/w4/4000-valid.json \

   # --problemset-path fresh_problems/stp/w4/50000-train.json \
    # --validset-path fresh_problems/stp/w4/4000-valid.json \
