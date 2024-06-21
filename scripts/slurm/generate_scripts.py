from pathlib import Path
template = '''#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=4G
#SBATCH --time=<time>
#SBATCH --array=<array>
#SBATCH --output=/scratch/tjhia/bilevin/slurm_outputs/<subdir>/<dom>/<agent>/%j.out

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

argfile=/scratch/tjhia/bilevin/scripts/slurm/<agent>_args.txt
args=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $argfile)
seed=$(echo $args | cut -d' ' -f1)
# chk=$(echo $args | cut -d' ' -f3)

# --checkpoint-path $chk \\

python bilevin/main.py \\
    --n-eval 32 \\
    --agent $agent \\
    --seed $seed \\
    --runsdir-path runs/<subdir>/<dom>/<agent> \\
    --problems-path problems/<dom>/50000-train.pkl \\
    --valid-path problems/<dom>/1000-valid.pkl \\
    \\
    --share-feature-net <sfn> \\
    --train-expansion-budget <budget> \\
'''

def generate_script(config_line):
    params = {}
    for pair in config_line.strip().split():
        key, value = pair.split('=')
        params[key] = value

    script_content = template
    for key, value in params.items():
        script_content = script_content.replace(f'<{key}>', value)

    filename = params['filename']
    with open(f"train_scripts/{filename}", 'w') as f:
        f.write(script_content)
    print(f'Script {filename} generated.')

def main():
    with open('train_configs.sh', 'r') as config_file:
        for line in config_file:
            line = line.strip()
            if line and not line.startswith('#'):
                generate_script(line)

if __name__ == '__main__':
    main()
