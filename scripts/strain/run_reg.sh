#!/bin/bash
# This runs parallel experiments over 10 seeds.
# USAGE EXAMPLE
# 	./run_reg.sh 1 1 0.1
# Run from this folder only.
NUM_SAMPLES=$1	# how many repetitions
NUM_CONC=$2		# number of concurrent tasks in the array job
RATIO=$3        # ratio of train data

LOG_DIR="${HOME}/logs/${MODEL}"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

for seed in {21..50}
do
	# submit to slurm
    sbatch \
    --array=1-${NUM_SAMPLES}%${NUM_CONC} \
    --output="${LOG_DIR}/${seed}-%A_%a.out" \
     ./reg.sh ${seed} ${RATIO}

done