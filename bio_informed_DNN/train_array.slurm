#!/bin/bash
#SBATCH --job-name=biomodel
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-71
#SBATCH --output=logs/slurm_%A_%a.out

module load CUDA
source ../venv311/bin/activate

# Hyperparameter grid
LR_LIST=(1e-4 3e-4 1e-3 3e-3)
WD_LIST=(0 1e-5 1e-4)
DROP_LIST=(0.0 0.1 0.2)
ACT_LIST=(relu gelu)

LR=${LR_LIST[$((SLURM_ARRAY_TASK_ID % 4))]}
WD=${WD_LIST[$(((SLURM_ARRAY_TASK_ID / 4) % 3))]}
DROP=${DROP_LIST[$(((SLURM_ARRAY_TASK_ID / 12) % 2))]}
ACT=${ACT_LIST[$(((SLURM_ARRAY_TASK_ID / 24) % 2))]}

python data_training.py \
    --learning-rate $LR \
    --weight-decay $WD \
    --dropout $DROP \
    --activation $ACT \
    --batch-size 512