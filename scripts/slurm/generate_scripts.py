from pathlib import Path

template = """#!/bin/bash
#SBATCH --account=def-lelis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=<mem>
#SBATCH --time=<time>
#SBATCH --array=7,17,31,53,97
#SBATCH --output=/scratch/tjhia/bilevin/outputs_bfs/<dom>-50000-train_<agent>_%A-%a-%j.out

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
    --weight-astar 2.5 \\
    --agent <agent> \\
    --seed $SLURM_ARRAY_TASK_ID \\
    --runsdir-path runs_bfs/<dom>/<agent> \\
    --train-path problems/<dom>/50000-train.pkl \\
    --valid-path problems/<dom>/1000-valid.pkl \\
    --test-path problems/<dom>/1000-test.pkl \\
    --slow-problem 30 \\
    --shuffle \\
    \\
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

    fname = params["fname"]
    with open(f"train_scripts/{fname}", "w") as f:
        f.write(script_content)
    print(f"Script {fname} generated.")


def main():
    with open("configs_tight.sh", "r") as config_file:
        for line in config_file:
            line = line.strip()
            if line and not line.startswith("#"):
                generate_script(line)


# if [[ $SLURM_ARRAY_TASK_ID -eq 7 ]]; then
#     chk=
# elif [[ $SLURM_ARRAY_TASK_ID -eq 17  ]]; then
#     chk=
# elif [[ $SLURM_ARRAY_TASK_ID -eq 31  ]]; then
#     chk=
# elif [[ $SLURM_ARRAY_TASK_ID -eq 53  ]]; then
#     chk=
# elif [[ $SLURM_ARRAY_TASK_ID -eq 97  ]]; then
#     chk=
# fi

if __name__ == "__main__":
    main()
