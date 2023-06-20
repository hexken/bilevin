#!/bin/bash

if [ "$1" == "r" ]; then
	rm args.txt
fi

touch args.txt

ebudgets="2000"
tbudgets="300"

for dirname in ../../runs/sok_t/Sokoban*; do
	for ebudget in $ebudgets; do
		for tbudget in $tbudgets; do
			tmp=${dirname#*Sokoban-}
	    		expname=${tmp%_167*}
			modelpath=runs/sok_t/$(basename $dirname)

			if [[ "$modelpath" == *BiLevin* ]]; then
				agent="BiLevin"
			else	
				agent="Levin"
			fi
		        echo $agent $ebudget $tbudget $modelpath $expname >> args.txt
		done
	done
done

