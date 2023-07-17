#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=186G
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/stpw4/%j.txt
#SBATCH --account=rrg-lelis

source $HOME/bilevin-env2/bin/activate
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
    --batch-size-train 40 \
    --mode train \
    --loss-fn cross_entropy_loss \
    --cost-fn levin_cost \
    --n-subgoals $5 \
    --grad-steps 10 \
    --feature-net-lr 0.001 \
    --forward-feature-net-lr 0.001 \
    --backward-feature-net-lr 0.001\
    --forward-policy-lr 0.001 \
    --backward-policy-lr 0.001 \
    --bootstrap-epochs 0 \
    --curriculum-epochs 5 \
    --permutation-epochs 100 \
    --epoch-reduce-lr 1000 \
    --epoch-reduce-grad-steps 1000 \
    --epoch-begin-validate 1 \
    --time-budget 300 \
    --agent $1 \
    --problemset-path $2 \
    --validset-path $3 \
    --expansion-budget $4 \
    --seed $6 \
    --wandb-mode disabled \
    --runsdir-path runs/stpw4 \
    --exp-name s_ce_lc_bg

