usage() {
	echo "usage: <dir> r|m"
	exit
}

if [[ $# -ne 2 ]]; then
	usage
fi

if [[ $2 == "r" ]]; then
	cmd="rm -rf"
elif [[ $2 == "m" ]]; then
	cmd="mkdir -p"
else
	usage
	exit
fi

eval ${cmd} ~/bilevin/slurm_outputs/${1}
eval ${cmd} ~/bilevin/runs/${1}
ls -d ~/bilevin/slurm_outputs/${1}
ls -d ~/bilevin/runs/${1}
