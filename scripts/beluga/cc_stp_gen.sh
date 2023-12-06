#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/%j.out

source $HOME/bilevin-env2/bin/activate

cd /scratch/tjhia/bilevin
./scripts/stp_gen.sh