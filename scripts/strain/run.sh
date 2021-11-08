#!/bin/bash
# This runs parallel experiments over all datasets.
# USAGE EXAMPLE
# 	./run.sh 1 1 classes.txt
# Run from this folder only.
NUM_SAMPLES=$1	# how many repetitions
NUM_CONC=$2		# number of concurrent tasks in the array job
RATIO=$3        # ratio of train data
DATASET_FILE=$4	# file with dataset list


LOG_DIR="${HOME}/logs/${MODEL}"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

while read d; do
	# submit to slurm
    sbatch \
    --array=1-${NUM_SAMPLES}%${NUM_CONC} \
    --output="${LOG_DIR}/${d}-%A_%a.out" \
     ./missing.sh ${d} ${RATIO}

done < ${DATASET_FILE}