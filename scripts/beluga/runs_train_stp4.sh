#!/bin/bash
set -euf -o pipefail
cd ~/scratch/bilevin

problemset=problems/stp4/50000-train.pkl
validset=problems/stp4/1000-valid.pkl


agents="BiLevin Levin"
exp_budgets="16000"
seeds="713"

for agent in $agents; do
	for budget in $exp_budgets; do
		for seed in $seeds; do
			sbatch scripts/beluga/cc_train_stp4.sh $agent $problemset $validset $budget $seed
		done
	done
done
	
