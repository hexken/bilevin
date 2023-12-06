#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/%j.out

source $HOME/bilevin-env2/bin/activate

cd /scratch/tjhia/bilevin
./scripts/cube3_gen.sh
