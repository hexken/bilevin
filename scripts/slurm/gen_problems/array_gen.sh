#!/bin/bash

optimizers=("SGD" "Adam")
lrs=(0.0001 0.0005 0.001 0.005)
nesterov=(t f)
momentum=(0.0 0.9)
losses=("cross_entropy_mse_loss" "levin_sum_mse_loss" "levin_avg_mse_loss")
seeds=(1 2 3)
max_grad_norms=(-1.0 1.0 2.0 5.0)

argfile="array_args.txt"
for seed in "${seeds[@]}"; do
    for mn in "${max_grad_norms[@]}"; do
        for opt in "${optimizers[@]}"; do
            for lr in "${lrs[@]}"; do
                for mom in "${momentum[@]}"; do
                    for loss in "${losses[@]}";do
                        if [ "$opt" == "SGD" ]; then
                            if [ $mom == "0.9" ]; then
                                for nest in "${nesterov[@]}"; do
                                    echo "$seed $opt $lr $mn $mom $loss $nest" >> $argfile
                                done
                            else
                                echo "$seed $opt $lr $mn 0.0 $loss f" >> $argfile
                            fi
                        elif [ "$mom" == "0.9" ]; then
                            continue
                        else
                            echo "$seed $opt $lr $mn $mom $loss f" >> $argfile
                        fi
                    done
                done
            done
        done
    done
done
