#!/bin/bash
# This runs parallel experiments.
# Run from this folder only.
NUM_SAMPLES=$1	# how many repetitions
NUM_CONC=$2		# number of concurrent tasks in the array job
SCRIPT=$3       # the name of the script to run in parallel

LOG_DIR="${HOME}/logs/gvma/weighted_jaccard"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

for ratio in 0.05 0.1 0.2 0.5
do
    for seed in {1..30}
    do
        # submit to slurm
        sbatch \
        --array=1-${NUM_SAMPLES}%${NUM_CONC} \
        --output="${LOG_DIR}/seed=${seed}_ratio=${ratio}-%A_%a.out" \
        ./${SCRIPT}.sh ${seed} ${ratio}
    done
done