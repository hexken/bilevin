#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=186G
#SBATCH --array=1-2
#SBATCH --time=00:15:00
#SBATCH --exclusive
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/ap28/%j.txt
#SBATCH --account=rrg-lelis

source $HOME/bilevin-env/bin/activate
cd $SLURM_TMPDIR
pip freeze > requirements.txt
python -m venv env
deactivate
source env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

argfile=/scratch/tjhia/bilevin/scripts/beluga/args.txt
args=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $argfile)
agent=$(echo $args | cut -d' ' -f1)
ebudget=$(echo $args | cut -d' ' -f2)
tbudget=$(echo $args | cut -d' ' -f3)
modelpath=$(echo $args | cut -d' ' -f4)
expname=$(echo $args | cut -d' ' -f5)

cd /scratch/tjhia/bilevin
export OMP_NUM_THREADS=1

python src/main.py \
    --world-size 40 \
    --mode "test" \
    --agent $agent \
    --problemset-path problems/stp/april/3w/1000-test.json \
    --model-suffix "best_expanded" \
    --expansion-budget $ebudget \
    --time-budget $tbudget \
    --seed 1 \
    --wandb-mode offline \
    --model-path $modelpath \
    --exp-name $expname
