#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --partition=cpufast
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G

SEED=$1
RATIO=$2

module load Julia

julia --threads 10 ./calculate_dm.jl $SEED $RATIO