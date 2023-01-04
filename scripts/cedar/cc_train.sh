#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=187G
#SBATCH --time=0:05:00
#SBATCH --exclusive
#SBATCH --constrain=cascade
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/%j.txt
#SBATCH --account=def-lelis

source $HOME/bilevin-env/bin/activate
cd /scratch/tjhia/bilevin
./scripts/tr1.sh
