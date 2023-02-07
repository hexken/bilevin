#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=186G
#SBATCH --time=0:5:00
#SBATCH --exclusive
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/%j.txt
#SBATCH --account=rrg-lelis

source $HOME/bilevin-env/bin/activate
cd /scratch/tjhia/bilevin
export OMP_NUM_THREADS=1

python src/main.py \
    --world-size 32 \
    --mode train \
    --agent $1 \
    --loss levin_loss_sum \
    --problemset-path $2 \
    --initial-budget $3 \
    --grad-steps 10 \
    --batch-size-bootstrap 32 \
    --seed $4 \
    --wandb-mode offline
