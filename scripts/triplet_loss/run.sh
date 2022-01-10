#!/bin/bash
# This runs parallel experiments over 10 seeds.
# USAGE EXAMPLE
# 	./run_reg.sh 1 1 0.1
# Run from this folder only.

LOG_DIR="${HOME}/logs/gvma/triplets"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

for dist in SqEuclidean CosineDist
do
    for all in 0 1
    do
        for seed in {1..10}
        do
            # submit to slurm
            sbatch \
            --output="${LOG_DIR}/seed=${seed}_all=${all}_dist=${dist}-%A_%a.out" \
            ./triplet.sh ${seed} ${all} ${dist}
        done
    done
done