#!/bin/bash
cd ~/scratch/bilevin
problemset=problems/stp/5w/50000-original.json
agents="Levin BiLevin"
seeds="1 2"

for agent in $agents; do
	for seed in $seeds; do
	#	sbatch scripts/beluga/cc_train.sh $agent $problemset $seed
		sbatch scripts/beluga/cc_train_localenv.sh $agent $problemset $seed
	done
done
	
