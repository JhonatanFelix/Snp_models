#!/bin/bash
#SBATCH --job-name=biomodel
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-272
#SBATCH --output=logs/slurm_%A_%a.out

module load CUDA
source ../venv311/bin/activate

# Reduced Hyperparameter Grid
LR_LIST=(1e-2 1e-3 1e-4)
WD_LIST=(1e-4 1e-3 1e-2)
DROP_LIST=(0.2 0.4 0.5)
BATCH_LIST=(64 128 256)
ARCHI_LIST=("100" "50" "50 25")

# Sizes
n_lr=${#LR_LIST[@]}
n_wd=${#WD_LIST[@]}
n_drop=${#DROP_LIST[@]}
n_batch=${#BATCH_LIST[@]}
n_arch=${#ARCHI_LIST[@]}

ID=$SLURM_ARRAY_TASK_ID

# Decode hyperparameter indices
lr_i=$((ID % n_lr))
ID=$((ID / n_lr))

wd_i=$((ID % n_wd))
ID=$((ID / n_wd))

drop_i=$((ID % n_drop))
ID=$((ID / n_drop))

batch_i=$((ID % n_batch))
ID=$((ID / n_batch))

arch_i=$((ID % n_arch))

LR=${LR_LIST[$lr_i]}
WD=${WD_LIST[$wd_i]}
DROP=${DROP_LIST[$drop_i]}
BATCH=${BATCH_LIST[$batch_i]}
ARCHI=${ARCHI_LIST[$arch_i]}

echo "Running with:"
echo "LR=$LR WD=$WD DROP=$DROP BATCH=$BATCH ARCH=$ARCHI"

python data_training.py \
    --criterion HuberLoss \
    --layers $ARCHI \
    --learning-rate $LR \
    --weight-decay $WD \
    --dropout $DROP \
    --activation gelu \
    --batch-size $BATCH