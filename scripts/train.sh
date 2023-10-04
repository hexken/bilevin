#!/bin/bash
#
export OMP_NUM_THREADS=1
# export PYTORCH_JIT=0
# export PYTORCH_JIT_LOG_LEVEL=">>graph_fuser.cpp"

python src/main.py \
    --world-size 4 \
    --mode train \
    --agent BiLevin \
    --loss loop_levin_loss \
    --cost-fn levin_cost \
    --feature-net-lr 0.001 \
    --forward-feature-net-lr 0.001 \
    --backward-feature-net-lr 0.001 \
    --forward-policy-lr 0.001 \
    --backward-policy-lr 0.001 \
    --min-samples-per-stage 100 \
    --min-stage-solve-ratio 0.01 \
    --epochs 3 \
    --epoch-reduce-lr 99 \
    --epoch-reduce-grad-steps 99 \
    --epoch-begin-validate 1 \
    --validate-every 1 \
    --expansion-budget 10 \
    --time-budget 300 \
    --grad-steps 10 \
    --seed 1 \
    --runsdir-path runs/ \
    --problems-path latest_problems/cube3/1000-train.pkl \
    --valid-path latest_problems/cube3/100-valid.pkl \
    # --problemset-path fresh_problems/cube3/50000-train.json \
    # --validset-path fresh_problems/cube3/4000-valid.json \
    # --problemset-path fresh_problems/stp/w3/5000-train.json \
    # --validset-path fresh_problems/stp/w3/400-valid.json \

    # --problemset-path fresh_problems/wit/w6c4/50000-train.json \
    # --validset-path fresh_problems/wit/w6c4/4000-valid.json \

    # --problemset-path fresh_problems/stp/w4/50000-train.json \
    # --validset-path fresh_problems/stp/w4/4000-valid.json \

   # --problemset-path fresh_problems/stp/w4/50000-train.json \
    # --validset-path fresh_problems/stp/w4/4000-valid.json \
