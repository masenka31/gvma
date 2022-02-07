#!/bin/bash
# This runs parallel experiments over 10 seeds.
# USAGE EXAMPLE
# 	./run.sh
# Run from this folder only.

LOG_DIR="${HOME}/logs/gvma/triplet_embedding"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

for seed in {1..5}
do
    for known in 5
    do
        for full in 0 1
        do
            for clean in 0 1
            do
                for margin in 1 10
                do
                    for batchsize in 128 256
                    do
                        # submit to slurm
                        sbatch \
                        --output="${LOG_DIR}/seed=${seed}_known=${known}_full=${full}_clean=${clean}_margin=${margin}_batchsize=${batchsize}-%A_%a.out" \
                        ./triplet_unknown.sh ${seed} ${known} ${full} ${clean} ${margin} ${batchsize}
                    done
                done
            done
        done
    done
done

for seed in {1..5}
do
    for known in 10 15
    do
        for full in 1
        do
            for clean in 0 1
            do
                for margin in 1 10
                do
                    for batchsize in 128 256
                    do
                        # submit to slurm
                        sbatch \
                        --output="${LOG_DIR}/seed=${seed}_known=${known}_full=${full}_clean=${clean}_margin=${margin}_batchsize=${batchsize}-%A_%a.out" \
                        ./triplet_unknown.sh ${seed} ${known} ${full} ${clean} ${margin} ${batchsize}
                    done
                done
            done
        done
    done
done