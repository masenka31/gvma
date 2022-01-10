#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=cpufast
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G

SEED=$1

module load Julia

julia ./run_weighted.jl $SEED