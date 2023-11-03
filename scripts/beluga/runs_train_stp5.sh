#!/bin/bash
set -euf -o pipefail
cd ~/scratch/bilevin

problemset=problems/stp5/50000-train.pkl
validset=problems/stp5/1000-valid.pkl


agents="BiLevin Levin"
exp_budgets="64000"
seeds="1332"

for agent in $agents; do
	for budget in $exp_budgets; do
		for seed in $seeds; do
			sbatch scripts/beluga/cc_train_stp5.sh $agent $problemset $validset $budget $seed
		done
	done
done
	
