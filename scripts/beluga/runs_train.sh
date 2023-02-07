#!/bin/bash
cd ~/scratch/bilevin
problemset=problems/stp/3w/50000.json
agents="BiLevin"
initial_budgets="500"
seeds="1 2"

for agent in $agents; do
	for budget in $initial_budgets; do
		for seed in $seeds; do
		#	sbatch scripts/beluga/cc_train.sh $agent $problemset $budget $seed
			sbatch scripts/beluga/cc_train_localenv.sh $agent $problemset $budget $seed
		done
	done
done
	
