#!/bin/bash
# This runs parallel experiments over 10 seeds.
# USAGE EXAMPLE
# 	./run.sh
# Run from this folder only.

LOG_DIR="${HOME}/logs/gvma/toy_problem/triplet"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

for seed in {1..5}
do
    for lambda in 30 60 90
    do
        for n_classes in 15 20 30 50
        do
            for unq in 0 1
            do
                for activation in swish relu
                do
                    # submit to slurm
                    sbatch \
                    --output="${LOG_DIR}/seed=${seed}-%A_%a.out" \
                    ./triplet.sh ${seed} ${unq} ${n_classes} ${lambda} ${activation}
                done
            done
        done
    done
done