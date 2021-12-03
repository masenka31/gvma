#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=cpufast
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G

SEED=$1
RATIO=$2
ALPHA=$3

module load Julia

julia ./new_reg_alpha.jl $SEED $RATIO $ALPHA $CLASS