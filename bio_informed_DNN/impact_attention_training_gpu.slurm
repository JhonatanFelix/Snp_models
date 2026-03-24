#!/bin/bash
#SBATCH --job-name=impact_attn
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/home/ulg/genan/jrfelix/projects/Snp_models/bio_informed_DNN/logs/impact_attention_%A_%a.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/ulg/genan/jrfelix/projects/Snp_models}
SCRIPT_DIR=${SCRIPT_DIR:-${PROJECT_ROOT}/bio_informed_DNN}
VENV_PATH=${VENV_PATH:-${PROJECT_ROOT}/venv311}
cd "${SCRIPT_DIR}"

module load CUDA
source "${VENV_PATH}/bin/activate"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export DISABLE_PANDERA_IMPORT_WARNING=True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TRAIT=${TRAIT:-2}
EPOCHS=${EPOCHS:-150}
CRITERION=${CRITERION:-HuberLoss}
ACTIVATION=${ACTIVATION:-gelu}
EARLY_STOP=${EARLY_STOP:-true}
SAVE_MODEL=${SAVE_MODEL:-true}
SEED=${SEED:-43}
GRAD_CLIP=${GRAD_CLIP:-1.0}
ANNOTATION_PATH=${ANNOTATION_PATH:-./data/preprocessed/vep_ingenes_newimpact.csv}
GENOTYPE_PREFIX=${GENOTYPE_PREFIX:-../data/ML/BBB2023_MD}
PHENOTYPE_PATH=${PHENOTYPE_PATH:-../data/ML/pheno_2023bbb_0twins_6traits_mask}
IMPACT_EMBED_DIM=${IMPACT_EMBED_DIM:-16}

if [[ -z "${NUM_WORKERS:-}" ]]; then
    if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${SLURM_CPUS_PER_TASK}" -gt 2 ]]; then
        NUM_WORKERS=$((SLURM_CPUS_PER_TASK - 2))
    else
        NUM_WORKERS=4
    fi
fi

LR_LIST=(1e-4 2e-4)
WD_LIST=(1e-4)
ATTN_DROP_LIST=(0.05 0.0)
DROP_LIST=(0.5)
BATCH_LIST=(128)
ARCHI_LIST=("50" "100")

n_lr=${#LR_LIST[@]}
n_wd=${#WD_LIST[@]}
n_attn=${#ATTN_DROP_LIST[@]}
n_drop=${#DROP_LIST[@]}
n_batch=${#BATCH_LIST[@]}
n_arch=${#ARCHI_LIST[@]}
num_configs=$((n_lr * n_wd * n_attn * n_drop * n_batch * n_arch))

task_id=${SLURM_ARRAY_TASK_ID:-0}

if (( task_id < 0 || task_id >= num_configs )); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${task_id}. Expected 0..$((num_configs - 1))." >&2
    exit 1
fi

id=${task_id}
lr_i=$((id % n_lr))
id=$((id / n_lr))
wd_i=$((id % n_wd))
id=$((id / n_wd))
attn_i=$((id % n_attn))
id=$((id / n_attn))
drop_i=$((id % n_drop))
id=$((id / n_drop))
batch_i=$((id % n_batch))
id=$((id / n_batch))
arch_i=$((id % n_arch))

LR=${LR_LIST[$lr_i]}
WD=${WD_LIST[$wd_i]}
ATTN_DROP=${ATTN_DROP_LIST[$attn_i]}
DROP=${DROP_LIST[$drop_i]}
BATCH=${BATCH_LIST[$batch_i]}
ARCHI=${ARCHI_LIST[$arch_i]}
read -r -a ARCHI_VALUES <<< "${ARCHI}"

CMD=(
    python3 data_training_impact_attention.py
    --trait "${TRAIT}"
    --criterion "${CRITERION}"
    --activation "${ACTIVATION}"
    --epochs "${EPOCHS}"
    --learning-rate "${LR}"
    --weight-decay "${WD}"
    --dropout "${DROP}"
    --attention-dropout "${ATTN_DROP}"
    --impact-embedding-dim "${IMPACT_EMBED_DIM}"
    --batch-size "${BATCH}"
    --seed "${SEED}"
    --early-stop "${EARLY_STOP}"
    --save-model "${SAVE_MODEL}"
    --grad-clip "${GRAD_CLIP}"
    --num-workers "${NUM_WORKERS}"
    --annotation-path "${ANNOTATION_PATH}"
    --genotype-prefix "${GENOTYPE_PREFIX}"
    --phenotype-path "${PHENOTYPE_PATH}"
    --layers
    "${ARCHI_VALUES[@]}"
)

echo "Job ID: ${SLURM_JOB_ID:-no_slurm_job_id}"
echo "Array task: ${task_id}/${num_configs}"
echo "Working directory: ${SCRIPT_DIR}"
echo "Python: $(command -v python3)"
echo "Trait=${TRAIT} Criterion=${CRITERION} Activation=${ACTIVATION}"
echo "Annotation=${ANNOTATION_PATH}"
echo "LR=${LR} WD=${WD} Dropout=${DROP} AttnDrop=${ATTN_DROP} Batch=${BATCH} EmbedDim=${IMPACT_EMBED_DIM} Arch=${ARCHI}"
echo "GradClip=${GRAD_CLIP} SaveModel=${SAVE_MODEL} NumWorkers=${NUM_WORKERS}"

nvidia-smi || true

srun --cpu-bind=cores "${CMD[@]}"
