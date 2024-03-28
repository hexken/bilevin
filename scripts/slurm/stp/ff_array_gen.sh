#!/bin/bash

seeds="1 2"
n_landmarkss="4 8 16"
n_batch_expansionss="8 16 32"
ffws="1 2 5"
argfile="ff_args.txt"

for seed in $seeds; do
    for n_landmarks in $n_landmarkss; do
        for n_batch_expansions in $n_batch_expansionss; do
            if [[ "$n_landmarks" -gt "$n_batch_expansions" ]]; then
                continue
            fi
            for ffw in $ffws; do
                echo "$seed ApproxFF $n_landmarks $n_batch_expansions $ffw" >> $argfile
            done
        done
    done
done
