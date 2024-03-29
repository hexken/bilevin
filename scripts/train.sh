#!/bin/bash

export OMP_NUM_THREADS=1

lr=0.0001
python src/bilevin/main.py \
    --exp-name "" \
    --runsdir-path runs/ \
    --problems-path problems/col5/50000-train.pkl \
    --valid-path problems/col5/1000-valid.pkl \
    --master-port 34568 \
    --seed 1 \
    --world-size 4 \
    --n-landmarks 16 \
    --n-batch-expansions 32 \
    --feature-net-type conv \
    --share-feature-net f\
    --no-feature-net f \
    --adj-consistency f \
    --adj-weight 5 \
    --ends-consistency \
    --ends-weight 5 \
    --children-weight 5 \
    --n-samples 10 \
    --samples-weight 5 \
    --mode train \
    --agent  AStar \
    --weight-astar 2.5 \
    --weight-mse-loss 0.1 \
    --loss-fn default \
    --max-grad-norm 1.0 \
    --optimizer Adam \
    --grad-steps 10 \
    \
    --num-kernels 32 \
    --kernel-size 1 2 \
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
    --stage-begin-validate 1 \
    --validate-every-n-stage 10 \
    --validate-every-epoch \
    --checkpoint-every-n-batch 100 \
    \
    --time-budget 300 \
    --train-expansion-budget 8000 \
    \
    --n-final-stage-epochs 5 \
    \
    --n-batch-tail 10 \
    \
    # --batch-begin-validate 1 \
    # --validate-every-n-batch 50 \
    # --min-batches-per-stage -1 \
    # --max-batches-per-stage -1 \
    # --min-batches-final-stage -1 \
    # --max-batches-final-stage -1 \
    # --min-solve-ratio-stage 0 \
    # --min-solve-ratio-exp 0 \
    # \
