#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --partition=cpufast
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

SEED=$1
RATIO=$2

module load Julia

julia ./reg_test.jl $SEED $RATIO