from pathlib import Path

template = """#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=6G
#SBATCH --time=<time>
#SBATCH --array=<array>
#SBATCH --output=/scratch/tjhia/bilevin/outputs/<dom>-50000-train_<agent>_%A-%a-%j.out

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

python bilevin/main.py \\
    --world-size 8 \\
    --batch-size 32 \\
    --n-eval 32 \\
    --agent <agent> \\
    --seed $SLURM_ARRAY_TASK_ID \\
    --runsdir-path runs/<dom>/<agent> \\
    --train-path problems/<dom>/50000-train.pkl \\
    --valid-path problems/<dom>/1000-valid.pkl \\
    --test-path problems/<dom>/1000-test.pkl \\
    --slow-problem 10 \\
    --shuffle \\
    \\
    --share-feature-net <sfn> \\
    --train-expansion-budget <budget> \\
"""


def generate_script(config_line):
    params = {}
    for pair in config_line.strip().split():
        key, value = pair.split("=")
        params[key] = value

    script_content = template
    for key, value in params.items():
        script_content = script_content.replace(f"<{key}>", value)

    filename = params["filename"]
    with open(f"train_scripts/{filename}", "w") as f:
        f.write(script_content)
    print(f"Script {filename} generated.")


def main():
    with open("configs.sh", "r") as config_file:
        for line in config_file:
            line = line.strip()
            if line and not line.startswith("#"):
                generate_script(line)


if __name__ == "__main__":
    main()
