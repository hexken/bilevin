#!/bin/bash

if [ "$1" == "r" ]; then
	rm args.txt
fi

touch args.txt

ebudgets="2000"
tbudgets="300"
dir=runs/sok_rand
puzzle=Sokoban

for dirname in ../../$dir/${puzzle}*; do
	for ebudget in $ebudgets; do
		for tbudget in $tbudgets; do
			tmp=${dirname#*${puzzle}-}
	    		expname=${tmp%_167*}
			modelpath=$dir/$(basename $dirname)

			if [[ "$modelpath" == *BiLevin* ]]; then
				agent="BiLevin"
			else	
				agent="Levin"
			fi
		        echo $agent $ebudget $tbudget $modelpath $expname >> args.txt
		done
	done
done

