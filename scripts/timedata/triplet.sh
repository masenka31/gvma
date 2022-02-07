#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

module load Julia

julia --threads 24 triplet_run.jl