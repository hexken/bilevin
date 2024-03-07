#!/bin/bash
#SBATCH --account=def-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=2G
#SBATCH --time=1:00:00
#SBATCH --array=1,6,11,16,21,26
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/thes/stp4/phs/%j.out

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

argfile=/scratch/tjhia/bilevin/scripts/slurm/stp/phs_args_test.txt
args=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $argfile)
agent=$(echo $args | cut -d' ' -f1)
model_path=$(echo $args | cut -d' ' -f2)
# chk=$(echo $args | cut -d' ' -f3)

    # --checkpoint-path $chk \

python src/bilevin/main.py \
    --agent $agent \
    --mode "test" \
    --world-size 4 \
    --seed 1 \
    --problems-path problems/stp4/1000-test.pkl \
    --exp-name "" \
