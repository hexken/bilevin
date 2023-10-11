#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=186G
#SBATCH --time=0:10:00
#SBATCH --exclusive
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/oct/cube3

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
    --epochs 10 \
    --epoch-reduce-lr 10000 \
    --epoch-reduce-grad-steps 10000 \
    --epoch-begin-validate 1 \
    --validate-every 1 \
    --time-budget 300 \
    --agent $1 \
    --problems-path $2 \
    --valid-path $3 \
    --expansion-budget $4 \
    --seed $5 \
    --share-feature-net \
    --runsdir-path runs/oct/cube3/
    #--min-samples-per-stage 5000 \
    #--min-stage-solve-ratio 0 \
