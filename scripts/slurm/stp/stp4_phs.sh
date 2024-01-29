#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=4G
#SBATCH --time=0:10:00
#SBATCH --array=13
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/stp4/%j.out

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

argfile=/scratch/tjhia/bilevin/scripts/slurm/stp/array_args.txt
args=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $argfile)
seed=$(echo $args | cut -d' ' -f1)
loss=$(echo $args | cut -d' ' -f2)
lr=$(echo $args | cut -d' ' -f3)
mn=$(echo $args | cut -d' ' -f4)


python src/bilevin/main.py \
    --agent PHS \
    --seed $seed \
    --runsdir-path runs/stp4 \
    --exp-name
    --problems-path new_problems/stp4/100000-train.pkl \
    --valid-path new_problems/stp4/1000-valid.pkl \
    --world-size 4 \
    --mode train \
    --max-grad-norm $mn \
    --loss-fn $loss \
    --grad-steps 10 \
    \
    --share-feature-net \
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
    --checkpoint-every-n-batch 200 \
    --validate-evory-epoch \
    \
    --time-budget 300 \
    --train-expansion-budget 2000 \
    \
    --min-batches-per-stage 800 \
    --min-solve-ratio-stage 0.95 \
    --min-solve-ratio-exp 0 \
    --n-final-stage-epochs 10 \
    \
    --n-batch-tail -1 \
    \
