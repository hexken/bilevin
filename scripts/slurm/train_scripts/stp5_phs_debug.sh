#!/bin/bash
#SBATCH --account=def-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=6G
#SBATCH --time=2-00:00
#SBATCH --array=1,3
#SBATCH --output=/scratch/tjhia/bilevin/outputs/stp5-50000-train_PHS_mask_%A-%a-%j.out

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
    --exp-name "mask" \
    --batch-size 32 \
    --n-eval 32 \
    --agent PHS \
    --seed $SLURM_ARRAY_TASK_ID \
    --runsdir-path runs/stp5/PHS \
    --slow-problem 10 \
    --train-path problems/stp5/50000-train.pkl \
    --valid-path problems/stp5/1000-valid.pkl \
    --test-path problems/stp5/1000-test.pkl \
    --shuffle \
    --mask-invalid-actions \
    \
    --share-feature-net True \
    --train-expansion-budget 7000 \
