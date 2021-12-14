#!/bin/bash
#SBATCH --time=2:30:00
#SBATCH --partition=amdfast
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

SEED=$1

module load Julia

julia ./toy_problem.jl $SEED