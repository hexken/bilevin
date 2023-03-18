#!/bin/bash
cd ~/scratch/bilevin
problemset=problems/stp/4w-cur/dset2/32000-train.json
validset=problems/stp/4w-cur/dset2/3200-valid.json
agents="Levin BiLevin"
initial_budgets="16000"
seeds="393 923"

for agent in $agents; do
	for budget in $initial_budgets; do
		for seed in $seeds; do
#			sbatch scripts/beluga/cc_train.sh $agent $problemset $validset $budget $seed
			sbatch scripts/beluga/cc_train_localenv.sh $agent $problemset $validset $budget $seed
		done
	done
done
	
