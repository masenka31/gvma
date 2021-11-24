#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=amdfast
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G

SEED=$1
RATIO=$2
ALPHA=$3
CLASS=$4

module load Julia

julia ./triplet.jl $SEED $RATIO $ALPHA $CLASS