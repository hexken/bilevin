#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --array=1,6
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/thes/tri5/phs/%j.out

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

argfile=/scratch/tjhia/bilevin/scripts/data/thes/tri5/phs_args_test.txt
args=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $argfile)
agent=$(echo $args | cut -d' ' -f1)
model_path=$(echo $args | cut -d' ' -f2)
# chk=$(echo $args | cut -d' ' -f3)

    # --checkpoint-path $chk \

python src/bilevin/main.py \
    --world-size 4 \
    --test-expansion-budget 7000 \
    --problems-path problems/tri5/1000-test.pkl \
    --mode "test" \
    --agent $agent \
    --model-path $model_path \
    --exp-name "" \
