#!/bin/bash

export OMP_NUM_THREADS=1

flr=0.0001
blr=0.0001
python src/bilevin/main.py \
    --agent  BiPHSBFS \
    --exp-name "" \
    --runsdir-path runs/ \
    --problems-path problems/stp4lt/1000-train.pkl \
    --valid-path problems/stp4/1000-valid.pkl \
    --master-port 34568 \
    --seed 1 \
    --world-size 4 \
    --n-landmarks 8 \
    --n-batch-expansions 32 \
    --feature-net-type conv \
    --share-feature-net t\
    --no-feature-net f \
    --adj-consistency f \
    --adj-weight 1 \
    --ends-consistency \
    --ends-weight 1 \
    --children-weight 1 \
    --n-samples 10 \
    --samples-weight 5 \
    --mode train \
    --weight-astar 2.5 \
    --weight-mse-loss 0.1 \
    --loss-fn default \
    --max-grad-norm 1.0 \
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
    --backward-heuristic-layers 256 298 128 \
    --backward-heuristic-lr $blr \
    \
    --stage-begin-validate 1 \
    --validate-every-n-stage 10 \
    --validate-every-epoch f\
    --checkpoint-every-n-batch 100 \
    \
    --time-budget 300 \
    --train-expansion-budget 2000 \
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
