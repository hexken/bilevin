#!/bin/bash
#SBATCH --account=def-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --array=138,330,522
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/stp5_phs/%j.out

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

argfile=/scratch/tjhia/bilevin/scripts/beluga/array_args.txt
args=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $argfile)
seed=$(echo $args | cut -d' ' -f1)
opt=$(echo $args | cut -d' ' -f2)
lr=$(echo $args | cut -d' ' -f3)
mn=$(echo $args | cut -d' ' -f4)
mom=$(echo $args | cut -d' ' -f5)
loss=$(echo $args | cut -d' ' -f6)
nest=$(echo $args | cut -d' ' -f7)
expname=opt${opt}_lr${lr}_n${nest}_mn${mn}_m${mom}_loss${loss}


python src/main.py \
    --agent PHS \
    --seed $seed \
    --runsdir-path runs/st5_phs \
    --exp-name $expname \
    --backend gloo \
    --problems-path problems/stp5c/120000-train.pkl \
    --valid-path problems/stp5c/1000-valid.pkl \
    --world-size 4 \
    --mode train \
    --loss-fn $loss \
    --optimizer $opt \
    --nesterov $nest \
    --momentum $mom \
    --max-grad-norm $mn \
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
    --batch-begin-validate 1 \
    --validate-every 800 \
    --checkpoint-every 200 \
    \
    --time-budget 300 \
    --train-expansion-budget 7000 \
    --max-expansion-budget -1 \
    --test-expansion-budget -1 \
    \
    --min-problems-per-stage -1 \
    --min-solve-ratio-stage 0 \
    --min-solve-ratio-exp 0 \
    --n-final-stage-epochs 10 \
    \
    --n-tail 0 \
    \
