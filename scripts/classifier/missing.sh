#!/bin/bash
#SBATCH --time=0:20:00
#SBATCH --partition=cpufast
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

CLASS=$1
RATIO=$2

module load Julia

julia ./class_missing_classifier.jl $CLASS $RATIO