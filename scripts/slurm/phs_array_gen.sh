#!/bin/bash

agents="PHS"
losses="levin_avg levin_sum nll_avg nll_sum"
mn="-1.0 1.0"
seeds="13 17 29"
argfile="phs_args.txt"

for seed in $seeds; do
    for agent in $agents; do
        for loss in $losses; do
            for m in $mn; do
                echo "$seed $agent $loss $m" >> $argfile
            done
        done
    done
done
