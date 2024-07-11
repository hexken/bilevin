#!/bin/bash
#SBATCH --account=def-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=6G
#SBATCH --time=56:00:00
#SBATCH --array=7,17,31,53,97
#SBATCH --output=/scratch/tjhia/bilevin/outputs/tri5-50000-train_BiPHS_%A-%a-%j.out

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

python bilevin/main.py \
    --world-size 8 \
    --batch-size 32 \
    --n-eval 32 \
    --agent BiPHS \
    --seed $SLURM_ARRAY_TASK_ID \
    --runsdir-path runs/tri5/BiPHS \
    --train-path problems/tri5/50000-train.pkl \
    --valid-path problems/tri5/1000-valid.pkl \
    --test-path problems/tri5/1000-test.pkl \
    --slow-problem 30 \
    --shuffle \
    \
    --share-feature-net False \
    --train-expansion-budget 4000 \
