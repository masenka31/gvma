#!/bin/bash
# This runs parallel experiments over 10 seeds.
# USAGE EXAMPLE
# 	./run_reg.sh 1 1 0.1
# Run from this folder only.
SCRIPT=$1       # the name of the script to run in parallel

LOG_DIR="${HOME}/logs/gvma/regularization"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

for ratio in 0.05 0.2 0.5
do
    for alpha in 0 0.1 1
    do
        for seed in {1..10}
        do
            # submit to slurm
            sbatch \
            --output="${LOG_DIR}/alpha=${alpha}_seed=${seed}_ratio=${ratio}-%A_%a.out" \
            ./${SCRIPT}.sh ${seed} ${ratio} ${alpha}
        done
    done
done