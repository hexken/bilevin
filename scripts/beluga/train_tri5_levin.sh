#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=186G
#SBATCH --time=12:00:00
#SBATCH --array=1-4
#SBATCH --exclusive
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/tri5_levin/%j.out

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

argfile=/scratch/tjhia/bilevin/scripts/beluga/levin_args.txt
args=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $argfile)
seed=$(echo $args | cut -d' ' -f1)
agent=$(echo $args | cut -d' ' -f2)
loss=$(echo $args | cut -d' ' -f3)
lr=$(echo $args | cut -d' ' -f4)
expname=lr${lr}


python src/main.py \
    --agent $agent \
    --seed $seed \
    --runsdir-path runs/tri5_levin \
    --exp-name $expname \
    --problems-path problems/wit_tri5/50000-train.pkl \
    --valid-path problems/wit_tri5/1000-valid.pkl \
    --world-size 40 \
    --mode train \
    --loss-fn $loss \
    --grad-steps 10 \
    \
    --share-feature-net \
    --num-kernels 32 \
    --kernel-size 1 2 \
    \
    --conditional-backward \
    \
    --forward-feature-net-lr $lr \
    --forward-policy-layers 128 \
    --forward-policy-lr $lr \
    --forward-heuristic-layers 128 \
    --forward-heuristic-lr $lr \
    \
    --backward-feature-net-lr $lr \
    --backward-policy-layers 256 198 128 \
    --backward-policy-lr $lr \
    --backward-heuristic-layers 256 198 128 \
    --backward-heuristic-lr $lr \
    \
    --batch-begin-validate 1 \
    --validate-every 125 \
    --checkpoint-every 50 \
    \
    --time-budget 10 \
    --train-expansion-budget 200000 \
    --max-expansion-budget 200000 \
    --test-expansion-budget 200000 \
    \
    --min-problems-per-stage -1 \
    --min-solve-ratio-stage 0 \
    --min-solve-ratio-exp 0 \
    --n-final-stage-epochs 3 \
    \
    --n-tail 0 \
    \
