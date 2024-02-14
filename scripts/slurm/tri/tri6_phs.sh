#!/bin/bash
#SBATCH --account=def-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=4G
#SBATCH --time=8:00:00
#SBATCH --array=1-10
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/socs/tri6/phs/%j.out

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

argfile=/scratch/tjhia/bilevin/scripts/slurm/stp/phs_args.txt
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
    --runsdir-path runs/socs/tri6/phs \
    --exp-name "" \
    --problems-path problems/tri6/125000-train.pkl \
    --valid-path problems/tri6/1000-valid.pkl \
    --world-size 4 \
    --mode train \
    --max-grad-norm 1.0 \
    --loss-fn traj_nll_mse_loss \
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
    --validate-every-n-stage 10 \
    --stage-begin-validate 5 \
    --checkpoint-every-n-batch 300 \
    --validate-every-epoch \
    \
    --time-budget 300 \
    --train-expansion-budget 2000 \
    \
    --min-batches-per-stage 1250 \
    --min-solve-ratio-stage 0.9 \
    --min-solve-ratio-exp 0 \
    --n-final-stage-epochs 20 \
    \
    --n-batch-tail 100 \
    \
