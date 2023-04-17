#!/bin/bash
cd ~/scratch/bilevin
#problemset=problems/witness/april/7w4c/50000-train.json
#validset=problems/witness/april/7w4c/4000-valid.json

#problemset=problems/sokoban/unfiltered/april/train.json
#validset=problems/sokoban/unfiltered/april/valid.json

problemset=problems/stp/april/3w/4000-train.json
validset=problems/stp/april/3w/1000-valid.json

agents="BiLevin Levin"
initial_budgets="500"
seeds="3"
#5 7 13 17"

for agent in $agents; do
	for budget in $initial_budgets; do
		for seed in $seeds; do
#			sbatch scripts/beluga/cc_train.sh $agent $problemset $validset $budget $epoch $seed
			sbatch scripts/beluga/cc_train_localenv.sh $agent $problemset $validset $budget $seed
		done
	done
done
	
