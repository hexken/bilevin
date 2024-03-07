#!/bin/bash
#SBATCH --account=def-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=1,6
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/thes/col5/astar/%j.out

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

argfile=/scratch/tjhia/bilevin/scripts/slurm/stp/astar_args.txt
args=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $argfile)
seed=$(echo $args | cut -d' ' -f1)
agent=$(echo $args | cut -d' ' -f2)
lr=0.0001
# chk=$(echo $args | cut -d' ' -f3)

    # --checkpoint-path $chk \

python src/bilevin/main.py \
    --agent $agent \
    --seed $seed \
    --weight-astar 2 \
    --weight-mse-loss 1 \
    --runsdir-path runs/thes/col5/astar \
    --exp-name "" \
    --problems-path problems/col5/300000-train.pkl \
    --valid-path problems/col5/1000-valid.pkl \
    --world-size 4 \
    --mode train \
    --max-grad-norm 1.0 \
    --loss-fn mse_loss \
    --grad-steps 10 \
    \
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
    --validate-every-n-stage 5 \
    --validate-every-epoch \
    --checkpoint-every-n-batch 300 \
    \
    --time-budget 300 \
    --train-expansion-budget 7000 \
    \
    --min-batches-per-stage 1250 \
    --max-batches-per-stage 2500 \
    --min-batches-final-stage 1250 \
    --min-solve-ratio-stage 0.9 \
    --n-final-stage-epochs 25 \
    \
    --n-batch-tail 100 \
    \
