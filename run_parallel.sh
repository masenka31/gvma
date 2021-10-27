#!/bin/bash
# This runs classifier on JSON on slurm.
# Run from this folder only.
NUM_SAMPLES=$1	# how many repetitions
NUM_CONC=$2	# number of concurrent tasks in the array job

LOG_DIR="${HOME}/logs/${CLASSIFIER}"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi


# submit to slurm
sbatch \
--array=1-${NUM_SAMPLES}%${NUM_CONC} \
--output="${LOG_DIR}/CLASSIFIER-%A_%a.out" \
    ./run_classifier.sh

# for local testing    
# ./${MODEL}_run.sh $MAX_SEED "MNIST" $METHOD 10