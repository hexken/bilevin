#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=187G
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/%j.txt
#SBATCH --account=def-lelis

source /home/tjhia/bilevin-env/bin/activate
cd /scratch/tjhia/bilevin
./scripts/tr3.sh
