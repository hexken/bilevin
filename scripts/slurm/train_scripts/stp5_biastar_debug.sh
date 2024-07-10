#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=6G
#SBATCH --time=00:10:00
#SBATCH --array=1
#SBATCH --output=/scratch/tjhia/bilevin/outputs/stp5-50000-train_BiAStar_%A-%a-%j.out

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
    --agent BiAStar \
    --seed $SLURM_ARRAY_TASK_ID \
    --runsdir-path runs/stp5/BiAStar \
    --train-path problems/stp5/50000-train.pkl \
    --valid-path problems/stp5/1000-valid.pkl \
    --test-path problems/stp5/1000-test.pkl \
    --slow-problem 10 \
    --shuffle \
    \
    --share-feature-net True \
    --train-expansion-budget 7000 \
