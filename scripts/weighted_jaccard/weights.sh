#!/bin/bash
#SBATCH --time=1:30:00
#SBATCH --partition=cpufast
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

SEED=$1
RATIO=$2

module load Julia

julia ./weighted_run.jl $SEED $RATIO