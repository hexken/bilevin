#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2G
#SBATCH --time=2:00:00
#SBATCH --array=4,5
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/%j.out

source $HOME/bilevin-env2/bin/activate

cd /scratch/tjhia/bilevin
./scripts/stp_gen.sh $SLURM_ARRAY_TASK_ID
