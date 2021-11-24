#!/bin/bash
# This runs parallel experiments over 10 seeds.
# USAGE EXAMPLE
# 	./submit.sh 1 1 classes.txt triplet
# Run from this folder only.
DATASET_FILE=$1	# file with dataset list
SCRIPT=$2       # the name of the script to run in parallel

LOG_DIR="${HOME}/logs/gvma/triplet_missing"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

while read d; do
    for ratio in 0.05 0.2 0.5
    do
        for alpha in 0 0.1 1
        do
            for seed in {1..10}
            do
                # submit to slurm
                sbatch \
                --output="${LOG_DIR}/alpha=${alpha}_seed=${seed}_ratio=${ratio}-%A_%a.out" \
                ./${SCRIPT}.sh ${seed} ${ratio} ${alpha} $d
            done
        done
    done
done < ${DATASET_FILE}