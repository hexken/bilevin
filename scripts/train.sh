#!/bin/bash

export OMP_NUM_THREADS=1

lr=0.0001
python src/bilevin/main.py \
    --exp-name "" \
    --runsdir-path runs/ \
    --problems-path problems/stp4nc/500-train.pkl \
    --valid-path problems/stp4nc/1000-valid.pkl \
    --master-port 34568 \
    --seed 1 \
    --world-size 4 \
    --mode train \
    --agent  BiPHSBFS \
    --weight-astar 2.5 \
    --weight-mse-loss 0.1 \
    --loss-fn traj_nll_mse_loss \
    --max-grad-norm 1.0 \
    --optimizer Adam \
    --grad-steps 10 \
    \
    --share-feature-net \
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
    --train-expansion-budget 10 \
    \
    --min-batches-per-stage -1 \
    --max-batches-per-stage -1 \
    --min-batches-final-stage -1 \
    --max-batches-final-stage -1 \
    --min-solve-ratio-stage 0.9 \
    --min-solve-ratio-exp 0 \
    --n-final-stage-epochs 10 \
    \
    --n-batch-tail 1 \
    \
    # --batch-begin-validate 1 \
    # --validate-every-n-batch 50 \
