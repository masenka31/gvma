#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --partition=cpufast
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

SEED=$1
RATIO=$2
ALPHA=$3

module load Julia

julia ./reg_test_alpha.jl $SEED $RATIO $ALPHA