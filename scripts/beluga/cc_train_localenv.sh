#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=186G
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/ap28/%j.txt
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
    --feature-net-lr 0.001 \
    --forward-policy-lr 0.001 \
    --backward-policy-lr 0.001 \
    --bootstrap-epochs 0 \
    --curriculum-epochs 20 \
    --permutation-epochs 0 \
    --include-prev-difficulty False \
    --permutation-focus False \
    --epoch-reduce-lr 1 \
    --epoch-reduce-grad-steps 99 \
    --epoch-begin-validate 1\
    --batch-size-train 40 \
    --time-budget 300 \
    --agent $1 \
    --problemset-path $2 \
    --validset-path $3 \
    --expansion-budget $4 \
    --seed $5 \
    --wandb-mode offline \
    --exp-name nomask
