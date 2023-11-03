#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=186G
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/nov/stp5/%j.out

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
    --mode train \
    --loss-fn levin_loss \
    --cost-fn levin_cost \
    --grad-steps 10 \
    --feature-net-lr 0.001 \
    --forward-hidden-layers 128 \
    --backward-hidden-layers 256 192 128 \
    --forward-feature-net-lr 0.001 \
    --backward-feature-net-lr 0.001\
    --forward-policy-lr 0.001 \
    --backward-policy-lr 0.001 \
    --batch-begin-validate 1 \
    --validate-every 10000 \
    --time-budget 300 \
    --agent $1 \
    --problems-path $2 \
    --valid-path $3 \
    --train-expansion-budget $4 \
    --test-expansion-budget 64000 \
    --increase-budget \
    --min-samples-per-stage 2500000 \
    --min-solve-ratio 0 \
    --n-solve-ratio 0 \
    --seed $5 \
    --share-feature-net \
    --runsdir-path runs/nov/stp5/
