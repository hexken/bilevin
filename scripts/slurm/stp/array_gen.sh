#!/bin/bash

lrs="0.0001 0.0005 0.001 0.005, 0.01"
losses="traj_nll_mse_loss levin_avg_mse_loss levin_sum_mse_loss"
seeds="1 2 3"
max_grad_norms="-1.0 0.001 0.005 0.01 0.5 1.0"

argfile="array_args.txt"
for seed in $seeds; do
    for loss in $losses; do
        for lr in $lrs; do
            for mn in $max_grad_norms; do
                echo "$seed $loss $lr $mn" >> $argfile
            done
        done
    done
done
