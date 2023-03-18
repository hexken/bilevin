#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=186G
#SBATCH --time=8:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/march17/%j.txt
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

python src/main.py \
    --world-size 40 \
    --mode train \
    --loss levin_loss \
    --grad-steps 10 \
    --learning-rate 0.001 \
    --bootstrap-epochs 1 \
    --curriculum-epochs 1 \
    --permutation-epochs 10 \
    --epochs-reduce-lr 1\
    --epoch-begin-validate 1\
    --batch-size-train 40 \
    --agent $1 \
    --problemset-path $2 \
    --validset-path $3 \
    --initial-budget $4 \
    --seed $5 \
    --wandb-mode offline \
    --exp-name "dset2"\	

