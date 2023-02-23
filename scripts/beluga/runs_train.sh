#!/bin/bash
cd ~/scratch/bilevin
problemset=problems/witness/7w4c/50000-train.json
validset=problems/witness/7w4c/1000-valid.json
agents="Levin BiLevin"
initial_budgets="16000"
epochs="5"
seeds="1 43"

for agent in $agents; do
	for budget in $initial_budgets; do
		for epoch in $epochs; do
			for seed in $seeds; do
	#			sbatch scripts/beluga/cc_train.sh $agent $problemset $validset $budget $epoch $seed
				sbatch scripts/beluga/cc_train_localenv.sh $agent $problemset $validset $budget $epoch $seed
			done
		done
	done
done
	
