#!/bin/bash
# This runs experiments

LOG_DIR="${HOME}/logs/time_gvma/triplet"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

# submit 20 experiments to slurm
for i in {1..20}
do
    # submit to slurm
    sbatch \
    --output="${LOG_DIR}/${i}_%A_%a.out" \
    ./triplet.sh
done