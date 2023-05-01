#!/bin/bash

if [ "$1" == "r" ]; then
	rm args.txt
fi

touch args.txt

ebudgets="500"
tbudgets="300"

for dirname in ../../runs/SlidingTile*; do
	for ebudget in $ebudgets; do
		for tbudget in $tbudgets; do
			tmp=${dirname#*SlidingTilePuzzle-}
	    		expname=${tmp%_167*}
			modelpath=runs/$(basename $dirname)

			if [[ "$modelpath" == *BiLevin* ]]; then
				agent="BiLevin"
			else	
				agent="Levin"
			fi
		        echo $agent $ebudget $tbudget $modelpath $expname >> args.txt
		done
	done
done

