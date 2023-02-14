#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=186G
#SBATCH --array=1-25
#SBATCH --time=30:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/%j.txt
#SBATCH --account=rrg-lelis

source $HOME/bilevin-env/bin/activate
cd $SLURM_TMPDIR
pip freeze > requirements.txt
python -m venv env
deactivate
source env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

args=$(sed "${SLURM_ARRAY_TASK_ID}q;d" args.txt)
modelpath=$(echo $args | cut -d' ' -f1)
expname=$(echo $args | cut -d' ' -f2)

cd /scratch/tjhia/bilevin
export OMP_NUM_THREADS=1

python src/main.py \
    --world-size 40 \
    --mode "test" \
    --agent Levin \
    --loss levin_loss_sum \
    --problemset-path problems/witness/4w4c/50000-original.json \
    --initial-budget 2000 \
    --seed 1 \
    --wandb-mode offline \
    --model-path $modelpath \
    --exp-name $expname

