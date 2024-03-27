#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=8G
#SBATCH --time=120:00:00
#SBATCH --array=1-10
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/thes/stp4nc/levin/%j.out

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

argfile=/scratch/tjhia/bilevin/scripts/slurm/stp/levin_args.txt
args=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $argfile)
seed=$(echo $args | cut -d' ' -f1)
agent=$(echo $args | cut -d' ' -f2)
lr=0.0001
# chk=$(echo $args | cut -d' ' -f3)

    # --checkpoint-path $chk \

python src/bilevin/main.py \
    --agent $agent \
    --seed $seed \
    --weight-mse-loss 0.1 \
    --runsdir-path runs/thes/stp4nc/levin \
    --exp-name "" \
    --problems-path problems/stp4nc/50000-train.pkl \
    --valid-path problems/stp4nc/1000-valid.pkl \
    --world-size 4 \
    --mode train \
    --max-grad-norm 1.0 \
    --loss-fn traj_nll_loss \
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
    --validate-every-epoch \
    --checkpoint-every-n-batch 100 \
    \
    --train-expansion-budget 2000 \
    \
    --n-final-stage-epochs 50 \
    \
    --n-batch-tail 100 \
    \
