#!/bin/bash
# This runs parallel experiments over 10 seeds.
# USAGE EXAMPLE
# 	./run_reg_missing.sh 1 1 0.1 Lamer
# Run from this folder only.
NUM_SAMPLES=$1	# how many repetitions
NUM_CONC=$2		# number of concurrent tasks in the array job
DATASET_FILE=$3	# file with dataset list

LOG_DIR="${HOME}/logs/gvma"

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
                    --array=1-${NUM_SAMPLES}%${NUM_CONC} \
                    --output="${LOG_DIR}/alpha=${alpha}_seed=${seed}_ratio=${ratio}-%A_%a.out" \
                    ./reg_missing.sh ${seed} ${ratio} ${alpha} $d

                

            done
        done
    done
done < ${DATASET_FILE}