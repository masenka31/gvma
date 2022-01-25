#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G

SEED=$1
UNQ=$2
N=$3
L=$4

module load Julia

julia ./run_weighted.jl $SEED $UNQ $N $L