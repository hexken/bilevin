#!/bin/bash

scripts_dir=~/bilevin/scripts/slurm/training_scripts
doms="stp tri col"
agents="levin bilevin astar biastar phs biphs"
sizes="4 5"

for dom in $doms; do
    for size in $sizes; do
        for agent in $agents; do
            script=$scripts_dir/${dom}${size}_${agent}.sh
            if [[ ! -f $script ]]; then
                echo "Script $script does not exist"
                continue
            fi
            if sbatch $script; then
                echo "Submitted $script"
            else
                echo "Failed to submit $script"
            fi
        done
    done
done
