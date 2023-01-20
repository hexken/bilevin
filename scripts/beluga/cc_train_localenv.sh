#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=186G
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/%j.txt
#SBATCH --account=rrg-lelis

source $HOME/bilevin-env/bin/activate
cd $SLURM_TMPDIR
pip freeze > requirements.txt
python -m venv env
deactivate
source env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt


cd /scratch/tjhia/bilevin
export OMP_NUM_THREADS=1

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=32 \
    --master_addr=$(hostname)\
    --master_port=34567 \
    src/main.py \
    --mode train \
    --agent $1 \
    --loss levin_loss_sum \
    --model-path trained_models/ \
    --problemset-path $2 \
    --initial-budget 10000 \
    --grad-steps 10 \
    --batch-size-bootstrap 32 \
    --seed $3 \
    --wandb-mode offline
