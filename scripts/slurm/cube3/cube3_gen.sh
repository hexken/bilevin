#!/bin/bash
#SBATCH --account=def-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --time=3:00:00
#SBATCH --array=3
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/%j.out

source $HOME/bilevin-env2/bin/activate

cd /scratch/tjhia/bilevin
./scripts/cube3_gen.sh $SLURM_ARRAY_TASK_ID
