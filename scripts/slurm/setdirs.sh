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
    echo "Removing directories for $1"
elif [[ $2 == "c" ]]; then
    mkdir -p ~/bilevin/slurm_outputs/${1}
    mkdir -p ~/bilevin/runs/${1}
    echo "Creating directories for $1"
elif [[ $2 == "r" ]]; then
    rm -rf ~/bilevin/slurm_outputs/${1}
    rm -rf ~/bilevin/runs/${1}
    mkdir -p ~/bilevin/slurm_outputs/${1}
    mkdir -p ~/bilevin/runs/${1}
    echo "Recreating directories for $1"
else
	usage
	exit
fi
