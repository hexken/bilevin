#!/bin/bash

seeds="1 2"
ffks="4 8 16"
ffbs="8 16 32"
ffws="1 2 5"
argfile="ff_args.txt"

for seed in $seeds; do
    for ffk in $ffks; do
        for ffb in $ffbs; do
            if [[ "$ffk" -gt "$ffb" ]]; then
                continue
            fi
            for ffw in $ffws; do
                echo "$seed ApproxFF $ffk $ffb $ffw" >> $argfile
            done
        done
    done
done
