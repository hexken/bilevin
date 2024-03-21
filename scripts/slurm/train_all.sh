#!/bin/bash

scripts_dir=~/bilevin/scripts/slurm
stp_dir=$scripts_dir/stp
pancakes_dir=$scripts_dir/pancake
tri_dir=$scripts_dir/tri
col_dir=$scripts_dir/col

agents="levin astar phs"
pancake_sizes="14 18 22"
stp_sizes="4 5"
tri_sizes="4 5"
col_sizes="4 5"

for agent in $agents; do
    for size in $pancake_sizes; do
        sbatch $pancakes_dir/pancake${size}_${agent}.sh
    done
done

for agent in $agents; do
    for size in $stp_sizes; do
        sbatch $stp_dir/stp${size}_${agent}.sh
    done
done

for agent in $agents; do
    for size in $tri_sizes; do
        sbatch $tri_dir/tri${size}_${agent}.sh
    done
done

for agent in $agents; do
    for size in $col_sizes; do
        sbatch $col_dir/col${size}_${agent}.sh
    done
done
