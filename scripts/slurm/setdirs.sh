usage() {
	echo "usage: <dir> d|c|r"
	exit
}

if [[ $# -ne 2 ]]; then
	usage
fi

if [[ $2 == "d" ]]; then
    rm -rf ~/bilevin/slurm_outputs/${1}
    rm -rf ~/bilevin/runs/${1}
elif [[ $2 == "c" ]]; then
    mkdir -p ~/bilevin/slurm_outputs/${1}
    mkdir -p ~/bilevin/runs/${1}
elif [[ $2 == "r" ]]; then
    rm -rf ~/bilevin/slurm_outputs/${1}
    rm -rf ~/bilevin/runs/${1}
    mkdir -p ~/bilevin/slurm_outputs/${1}
    mkdir -p ~/bilevin/runs/${1}
else
	usage
	exit
fi
