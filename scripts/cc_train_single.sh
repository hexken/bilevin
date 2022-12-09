#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=187G
#SBATCH --time=6:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/tjhia/bi-levin/slurm_outputs/%j.txt
#SBATCH --account=def-lelis

module purge

source /home/tjhia/bilevin-env/bin/activate
cd /scratch/tjhia/bi-levin
./scripts/test_levin_stp_torchrun.sh
