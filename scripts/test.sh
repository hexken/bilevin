#!/bin/bash
#
export OMP_NUM_THREADS=1

python src/bilevin/main.py \
    --world-size 4 \
    --mode "test" \
    --agent BiLevin \
    --problems-path problems/cube3/1000-test.pkl \
    --test-expansion-budget 4 \
    --seed 1 \
    --model-path runs/cube3-1100-train_BiLevin_e4_t300.0_1_1698863272/model_best_expanded.pt \
    --exp-name ""
