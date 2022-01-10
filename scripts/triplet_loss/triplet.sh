#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --partition=amdfast
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G

SEED=$1
ALL=$2
DIST=$3

module load Julia

julia ./run_triplet.jl $SEED $ALL $DIST