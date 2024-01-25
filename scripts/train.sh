#!/bin/bash

export OMP_NUM_THREADS=1

lr=0.0005
python src/bilevin/main.py \
    --exp-name "" \
    --runsdir-path runs/ \
    --problems-path new_problems/stp4/100000-train.pkl \
    --valid-path new_problems/stp4/1000-valid.pkl \
    --master-port 34567 \
    --seed 1 \
    --world-size 4 \
    --mode train \
    --agent PHS \
    --weight-astar 2.5 \
    --loss-fn levin_avg_mse_loss \
    --max-grad-norm 2.0 \
    --optimizer Adam \
    --grad-steps 10 \
    \
    --share-feature-net \
    --num-kernels 32 \
    --kernel-size 1 2 \
    \
    --conditional-backward \
    \
    --forward-feature-net-lr $lr \
    --forward-policy-layers 128 \
    --forward-policy-lr $lr \
    --forward-heuristic-layers 128 \
    --forward-heuristic-lr $lr \
    \
    --backward-feature-net-lr $lr \
    --backward-policy-layers 256 198 128 \
    --backward-policy-lr $lr \
    --backward-heuristic-layers 256 298 128 \
    --backward-heuristic-lr $lr \
    \
    --batch-begin-validate 1 \
    --validate-every-n-batch -1 \
    --stage-begin-validate 1 \
    --validate-every-n-stage 10 \
    --validate-every-epoch \
    --checkpoint-every-n-batch 100 \
    \
    --time-budget 300 \
    --train-expansion-budget 2000 \
    \
    --min-batches-per-stage 800 \
    --min-solve-ratio-stage 0.9 \
    --min-solve-ratio-exp 0 \
    --n-final-stage-epochs 5 \
    \
    --n-batch-tail 100 \
    \
