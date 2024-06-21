#!/bin/bash

agents="PHS"
losses="default"
mn="-1.0"
seeds="13 17 29 31 37"
argfile="phs_args.txt"

for seed in $seeds; do
    for agent in $agents; do
        echo "$seed $agent" >> $argfile
    done
done
