#!/bin/bash
set -euf -o pipefail
cd ~/scratch/bilevin

problemset=problems/cube3/50000-train.pkl
validset=problems/cube3/1000-valid.pkl


agents="BiLevin"
exp_budgets="21000"
seeds="313"
#717 1013"
#7865"

for agent in $agents; do
	for budget in $exp_budgets; do
		for seed in $seeds; do
			sbatch scripts/beluga/cc_train_localenv.sh $agent $problemset $validset $budget $seed
		done
	done
done
	
