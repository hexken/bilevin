#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=4G
#SBATCH --time=5:00:00
#SBATCH --array=1-576
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/stp3_phs/%j.out

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

argfile=/scratch/tjhia/bilevin/scripts/beluga/array_args.txt
args=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $argfile)
seed=$(echo $args | cut -d' ' -f1)
opt=$(echo $args | cut -d' ' -f2)
lr=$(echo $args | cut -d' ' -f3)
loss=$(echo $args | cut -d' ' -f6)


python src/bilevin/main.py \
    --agent PHS \
    --seed $seed \
    --runsdir-path runs/stp3_phs \
    --exp-name 
    --problems-path problems/stp3/17500-train.pkl \
    --valid-path problems/stp3/1000-valid.pkl \
    --world-size 4 \
    --mode train \
    --loss-fn $loss \
    --optimizer $opt \
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
    --validate-every 625 \
    --checkpoint-every 200 \
    \
    --time-budget 300 \
    --train-expansion-budget 500 \
    --max-expansion-budget -1 \
    --test-expansion-budget -1 \
    \
    --min-problems-per-stage -1 \
    --min-solve-ratio-stage 0 \
    --min-solve-ratio-exp 0 \
    --n-final-stage-epochs 10 \
    \
    --n-tail 0 \
    \
