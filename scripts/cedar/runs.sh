#!/bin/bash
cd ~/scratch/bilevin
problemset=problems/witness/4w4c/50000-original.json
agents="Levin"
initial_budgets="2000"
seeds="1 2"

for agent in $agents; do
	for budget in $initial_budgets; do
		for seed in $seeds; do
	#		sbatch scripts/cedar/cc_train.sh $agent $problemset $budget $seed
			sbatch scripts/cedar/cc_train_localenv.sh $agent $problemset $budget $seed
		done
	done
done
