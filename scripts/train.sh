#!/bin/bash

export OMP_NUM_THREADS=1

flr=0.0001
blr=0.0001
python bilevin/main.py \
    --agent PHS \
    --loss-fn nll_sum \
    --exp-name "" \
    --runsdir-path runs/ \
    --problems-path problems/lelis/stp5/50000-train.pkl \
    --valid-path problems/lelis/stp5/1000-valid.pkl \
    --master-port 34568 \
    --seed 1 \
    --world-size 4 \
    --n-eval 32 \
    --feature-net-type conv \
    --no-feature-net f \
    --share-feature-net f \
    --mode train \
    --weight-astar 2.5 \
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
    --validate-every-n-batch 10 \
    --validate-every-epoch f\
    --checkpoint-every-n-batch 100 \
    \
    --time-budget 300 \
    --train-expansion-budget 500 \
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
