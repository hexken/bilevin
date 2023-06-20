#!/bin/bash
cd ~/scratch/bilevin
#problemset=problems/witness/may/4w4c/50000-train.json
#validset=problems/witness/may/4w4c/4000-valid.json
#problemset=problems/witness/may/5w4c/50000-train.json
#validset=problems/witness/may/5w4c/4000-valid.json
#problemset=problems/witness/may/6w4c/50000-train.json
#validset=problems/witness/may/6w4c/4000-valid.json

#problemset=problems/sokoban/unfiltered/april/train.json
#validset=problems/sokoban/unfiltered/april/valid.json

#problemset=problems/stp/april/3w/4000-train.json
#validset=problems/stp/april/3w/1000-valid.json

#problemset=problems/stp/4w-simplecur/100000-train.json
#validset=problems/stp/4w-simplecur/4000-valid.json
#problemset=problems/stp/4w-debug/100000-train.json
#validset=problems/stp/4w-debug/10000-valid.json

#problemset=problems/stp/5w-june/50000-train.json
#validset=problems/stp/5w-june/4000-valid.json
problemset=problems/stp/5w-june/50000-train.json
validset=problems/stp/5w-june/4000-valid.json

#problemset=problems/stp/april/4w/50000-train.json
#validset=problems/stp/april/4w/4000-valid.json

#problemset=problems/stp/april/5w/50000-train.json
#validset=problems/stp/april/5w/4000-valid.json

agents="BiLevin Levin"
exp_budgets="64000"
seeds="313"
#7865"

for agent in $agents; do
	for budget in $exp_budgets; do
		for seed in $seeds; do
#			sbatch scripts/cedar/cc_train.sh $agent $problemset $validset $budget $epoch $seed
			sbatch scripts/cedar/cc_train_localenv.sh $agent $problemset $validset $budget $seed
		done
	done
done
	
