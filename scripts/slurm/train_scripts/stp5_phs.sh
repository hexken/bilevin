#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --array=1
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/phs/stp5/phs/%j.out

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

argfile=/scratch/tjhia/bilevin/scripts/slurm/phs_args.txt
args=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $argfile)
seed=$(echo $args | cut -d' ' -f1)
# chk=$(echo $args | cut -d' ' -f3)

# --checkpoint-path $chk \

python bilevin/main.py \
    --n-eval 32 \
    --agent $agent \
    --seed $seed \
    --runsdir-path runs/phs/stp5/phs \
    --problems-path problems/stp5/50000-train.pkl \
    --valid-path problems/stp5/1000-valid.pkl \
    \
    --share-feature-net True \
    --train-expansion-budget 8000 \
