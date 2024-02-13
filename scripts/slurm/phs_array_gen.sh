#!/bin/bash

lrs="0.0001"
agents="PHS"
losses="traj_nll_mse_loss"
seeds="1 2 3 4 5"
argfile="phs_array_args.txt"

for seed in $seeds; do
    for agent in $agents; do
        echo "$seed $agent" >> $argfile
    done
done
