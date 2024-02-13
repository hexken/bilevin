#!/bin/bash

agents="Levin"
losses="traj_nll_loss"
seeds="1 2 3 4 5"
argfile="levin_array_args.txt"

for seed in $seeds; do
    for agent in $agents; do
        echo "$seed $agent" >> $argfile
    done
done
