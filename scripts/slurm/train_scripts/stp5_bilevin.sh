#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=4096M
#SBATCH --time=00:10:00
#SBATCH --array=1
#SBATCH --output=/scratch/tjhia/bilevin/outputs/stp5-50000-train_BiLevin_%a_%j-%a.out

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
    --world-size 6 \
    --batch-size 32 \
    --n-eval 32 \
    --agent BiLevin \
    --seed $SLURM_ARRAY_TASK_ID \
    --runsdir-path runs/stp5/BiLevin \
    --train-path problems/stp5/50000-train.pkl \
    --valid-path problems/stp5/1000-valid.pkl \
    --test-path problems/stp5/1000-test.pkl \
    --shuffle \
    \
    --share-feature-net True \
    --train-expansion-budget 7000 \
