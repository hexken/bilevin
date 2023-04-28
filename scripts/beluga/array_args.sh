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
	    		expname=${tmp%_167*}_test_e${ebudget}_t${tbudget}
			modelpath=runs/$(basename $dirname)

			if [[ "$modelpath" == *BiLevin* ]]; then
				agent="BiLevin"
			else	
				agent="Levin"
			fi
		        echo $agent $modelpath $expname >> args.txt
		done
	done
done

