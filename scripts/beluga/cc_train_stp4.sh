#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=186G
#SBATCH --time=12:00:00
#SBATCH --array=1-4
#SBATCH --exclusive
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/stp4/%j.out

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

argfile=/scratch/tjhia/bilevin/scripts/beluga/algs_seeds_losses.txt
args=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $argfile)
seed=$(echo $args | cut -d' ' -f1)
agent=$(echo $args | cut -d' ' -f2)
loss=$(echo $args | cut -d' ' -f3)


python src/main.py \
    --agent $agent \
    --seed $seed \
    --runsdir-path runs/stp4 \
    --problems-path problems/stp4/50000-train.pkl \
    --valid-path problems/stp4/1000-valid.pkl \
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
    --forward-feature-net-lr 0.001 \
    --forward-policy-layers 128 \
    --forward-policy-lr 0.001 \
    --forward-heuristic-layers 128 \
    --forward-heuristic-lr 0.001 \
    \
    --backward-feature-net-lr 0.001 \
    --backward-policy-layers 256 198 128 \
    --backward-policy-lr 0.001 \
    --backward-heuristic-layers 256 298 128 \
    --backward-heuristic-lr 0.001 \
    \
    --batch-begin-validate 1 \
    --validate-every 125 \
    --checkpoint-every 10 \
    \
    --time-budget 300 \
    --train-expansion-budget 24000 \
    --max-expansion-budget 24000 \
    --test-expansion-budget 24000 \
    \
    --min-samples-per-stage 250000 \
    --min-solve-ratio-stage 0 \
    --min-solve-ratio-exp 0 \
    \
    --n-tail 1000 \
    \
