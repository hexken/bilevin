#!/bin/bash
cd ~/scratch/bilevin
problemset=problems/witness/4w4c/test-1000-original.json
agents="Levin"
initial_budgets="2000"
seeds="1"
model=runs/bugfixed/Witness-4w4c-50000-original_Levin-2000_1_1675411815

for agent in $agents; do
	for budget in $initial_budgets; do
		for seed in $seeds; do
			sbatch scripts/beluga/cc_test.sh $agent $problemset $budget $seed $model
#			sbatch scripts/beluga/cc_test_localenv.sh $agent $problemset $budget $seed $model
		done
	done
done
	
