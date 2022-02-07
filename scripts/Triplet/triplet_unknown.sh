#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=cpufast
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

seed=$1
known=$2
full=$3
clean=$4
margin=$5
batchsize=$6

module load Julia

julia ./triplet_unknown.jl $seed $known $full $clean $margin $batchsize