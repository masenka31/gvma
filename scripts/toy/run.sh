#!/bin/bash
# This runs parallel experiments over 10 seeds.
# USAGE EXAMPLE
# 	./run_reg.sh 1 1 0.1
# Run from this folder only.

LOG_DIR="${HOME}/logs/gvma/toy_problem"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

for seed in {1..10}
do
    # submit to slurm
    sbatch \
    --output="${LOG_DIR}/seed=${seed}-%A_%a.out" \
    ./toy.sh ${seed}
done
