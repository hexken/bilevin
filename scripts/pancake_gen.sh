#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR/../src/bilevin
cd $SCRIPT_DIR/../

if [ "$1" = "d" ]; then
    python src/bilevin/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  new_problems/pancaked/ \
    --seed 4238 \
    --width  10 \
    --random-goal \
    --randomize-test-steps \
    --min-steps 50 \
    --max-steps 200 \
    --n-train 100 \
    --n-valid  10 \
    --n-test 10
elif [ "$1" = "10rg" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  new_problems/pancake10rg/ \
    --seed 4238 \
    --width  10 \
    --random-goal \
    --randomize-test-steps \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 50000 \
    --n-valid  1000 \
    --n-test 1000
elif [ "$1" = "12rg" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  new_problems/pancake12rg/ \
    --seed 3316 \
    --width  12 \
    --random-goal \
    --randomize-test-steps \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 50000 \
    --n-valid  1000 \
    --n-test 1000
elif [ "$1" = "10" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  new_problems/pancake10/ \
    --seed 9298 \
    --width  10 \
    --permutation-test \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 50000 \
    --n-valid  1000 \
    --n-test 1000
elif [ "$1" = "12" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  new_problems/pancake12/ \
    --seed 1016 \
    --width  12 \
    --permutation-test \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 50000 \
    --n-valid  1000 \
    --n-test 1000
elif [ "$1" = "14rg" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  new_problems/pancake14rg/ \
    --seed 4238 \
    --width  14 \
    --random-goal \
    --randomize-test-steps \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 50000 \
    --n-valid  1000 \
    --n-test 1000
elif [ "$1" = "14" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  new_problems/pancake14/ \
    --seed 3316 \
    --width  14 \
    --permutation-test \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 50000 \
    --n-valid  1000 \
    --n-test 1000
elif [ "$1" = "16rg" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  new_problems/pancake16rg/ \
    --seed 4238 \
    --width  16 \
    --random-goal \
    --randomize-test-steps \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 50000 \
    --n-valid  1000 \
    --n-test 1000
elif [ "$1" = "20rg" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  new_problems/pancake20rg/ \
    --seed 3316 \
    --width  20 \
    --random-goal \
    --randomize-test-steps \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 50000 \
    --n-valid  1000 \
    --n-test 1000
elif [ "$1" = "16" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  new_problems/pancake16/ \
    --seed 9298 \
    --width  16 \
    --permutation-test \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 50000 \
    --n-valid  1000 \
    --n-test 1000
elif [ "$1" = "20" ]; then
    python bilevin/domains/puzzle_generator.py \
    --domain pancake \
    --output-path  new_problems/pancake20/ \
    --seed 1016 \
    --width  20 \
    --permutation-test \
    --min-steps 50 \
    --max-steps 1000 \
    --n-train 50000 \
    --n-valid  1000 \
    --n-test 1000
else
    echo "Invalid argument"
fi
