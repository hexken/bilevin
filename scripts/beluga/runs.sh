#!/bin/bash
cd ~/scratch/bilevin
problemset=problems/witness/6x6_5/50000.json
agent=Levin
seeds="1 2"

for seed in $seeds; do
	sbatch scripts/beluga/cc_train_localenv.sh $agent $problemset $seed
done
	
