#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

SEED=$1
UNQ=$2
N=$3
L=$4
ACT=$5

module load Julia

julia ./run_triplet.jl $SEED $UNQ $N $L $ACT