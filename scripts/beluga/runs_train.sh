#!/bin/bash
set -euf -o pipefail
cd ~/scratch/bilevin

problemset=fresh_problems/stp/w4/50000-train.json
validset=fresh_problems/stp/w4/4000-valid.json
#problemset=fresh_problems/stp/w4/50000-train.json
#validset=fresh_problems/stp/w4/4000-valid.json


agents="BiLevin"
exp_budgets="24000"
seeds="313 717 1013"
n_subgoals="50"
#7865"

for agent in $agents; do
	for budget in $exp_budgets; do
		for n in $n_subgoals; do
			for seed in $seeds; do
				sbatch scripts/beluga/cc_train_localenv.sh $agent $problemset $validset $budget $n $seed
			done
		done
	done
done
	
