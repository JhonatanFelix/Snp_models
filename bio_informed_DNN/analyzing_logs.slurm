#!/bin/bash
#SBATCH --job-name=biomodel
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=log_analysis_results/slurm_%A_%a.out

source ../venv311/bin/activate

python analyze_logs.py