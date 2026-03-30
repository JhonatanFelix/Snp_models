#!/bin/bash
# Compact resume_workflow smoke run for the two model families currently used most:
# - gene_only
# - attention_gene
#
# Purpose:
# - prove that the reduced reference workflow can instantiate, train, and save
#   both models with the current project data layout
# - keep defaults short enough for a quick validation run
#
# Notes:
# - The default configuration uses 5 epochs as a smoke test.
# - Set EPOCHS=150 (or another value) at submission time for a full run.

#SBATCH --job-name=resume_models
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/home/ulg/genan/jrfelix/projects/attention_tests/Snp_models/bio_informed_DNN/logs/resume_workflow_demo_%A_%a.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/ulg/genan/jrfelix/projects/attention_tests/Snp_models}
RESUME_DIR=${RESUME_DIR:-${PROJECT_ROOT}/bio_informed_DNN/resume_workflow}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-${RESUME_DIR}/train.py}
VENV_PATH=${VENV_PATH:-${PROJECT_ROOT}/venv_module_torch}
CUDA_MODULE=${CUDA_MODULE:-CUDA/12.1.1}
PYTORCH_MODULE=${PYTORCH_MODULE:-PyTorch/2.1.2-foss-2023a-CUDA-12.1.1}

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/bio_informed_DNN/logs"
mkdir -p "${RESUME_DIR}/.cache/matplotlib"

module purge
module load "${CUDA_MODULE}"
module load "${PYTORCH_MODULE}"
source "${VENV_PATH}/bin/activate"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export DISABLE_PANDERA_IMPORT_WARNING=True
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${RESUME_DIR}/.cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-${RESUME_DIR}/.cache/matplotlib}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TRAIT=${TRAIT:-4}
ACTIVATION=${ACTIVATION:-gelu}
CRITERION=${CRITERION:-HuberLoss}
OPTIMIZER=${OPTIMIZER:-adam}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
DROPOUT=${DROPOUT:-0.5}
ATTENTION_DROPOUT=${ATTENTION_DROPOUT:-0.0}
IMPACT_EMBED_DIM=${IMPACT_EMBED_DIM:-16}
BATCH_SIZE=${BATCH_SIZE:-128}
EPOCHS=${EPOCHS:-5}
VAL_SIZE=${VAL_SIZE:-1000}
SEED=${SEED:-43}
GRAD_CLIP=${GRAD_CLIP:-1.0}
EARLY_STOP=${EARLY_STOP:-true}
USE_LAYERNORM=${USE_LAYERNORM:-true}
SAVE_MODEL=${SAVE_MODEL:-true}
NUM_WORKERS=${NUM_WORKERS:-4}
OUTPUT_ROOT=${OUTPUT_ROOT:-${RESUME_DIR}/outputs_demo}

ANNOTATION_PATH=${ANNOTATION_PATH:-${PROJECT_ROOT}/data/ML/vep_ingenes_newimpact.csv}
GENOTYPE_PREFIX=${GENOTYPE_PREFIX:-${PROJECT_ROOT}/data/ML/BBB2023_MD}
PHENOTYPE_PATH=${PHENOTYPE_PATH:-${PROJECT_ROOT}/data/ML/pheno_2023bbb_0twins_6traits_mask}
FULL_PHENOTYPE_PATH=${FULL_PHENOTYPE_PATH:-${PROJECT_ROOT}/data/ML/pheno_20000bbb_6traits}
PEDIGREE_PATH=${PEDIGREE_PATH:-${PROJECT_ROOT}/data/ML/pedi_full_list.txt}
PATHWAY_MASK_PATH=${PATHWAY_MASK_PATH:-${PROJECT_ROOT}/data/preprocessed/Bos_taurus_new/mask_gene_pathway.npz}
PATHWAY_MAPPING_PATH=${PATHWAY_MAPPING_PATH:-${PROJECT_ROOT}/data/preprocessed/Bos_taurus_new/pathway_index_mapping.csv}
PATHWAY_GENE_MAPPING_PATH=${PATHWAY_GENE_MAPPING_PATH:-${PROJECT_ROOT}/data/preprocessed/Bos_taurus_new/gene_index_mapping.csv}

CONFIGS=(
    "gene_only|50|resume_gene_only_proof"
    "attention_gene|50|resume_attention_gene_proof"
)

task_id=${SLURM_ARRAY_TASK_ID:-0}
n_tasks=${#CONFIGS[@]}

if (( task_id < 0 || task_id >= n_tasks )); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${task_id}. Expected 0..$((n_tasks - 1))." >&2
    exit 1
fi

IFS='|' read -r MODEL_TYPE LAYERS LABEL <<< "${CONFIGS[$task_id]}"
read -r -a LAYER_VALUES <<< "${LAYERS}"

CMD=(
    python3 "${TRAIN_SCRIPT}"
    --model-type "${MODEL_TYPE}"
    --trait "${TRAIT}"
    --layers
    "${LAYER_VALUES[@]}"
    --annotation-path "${ANNOTATION_PATH}"
    --genotype-prefix "${GENOTYPE_PREFIX}"
    --phenotype-path "${PHENOTYPE_PATH}"
    --full-phenotype-path "${FULL_PHENOTYPE_PATH}"
    --pedigree-path "${PEDIGREE_PATH}"
    --pathway-mask-path "${PATHWAY_MASK_PATH}"
    --pathway-mapping-path "${PATHWAY_MAPPING_PATH}"
    --pathway-gene-mapping-path "${PATHWAY_GENE_MAPPING_PATH}"
    --activation "${ACTIVATION}"
    --criterion "${CRITERION}"
    --optimizer "${OPTIMIZER}"
    --learning-rate "${LEARNING_RATE}"
    --weight-decay "${WEIGHT_DECAY}"
    --dropout "${DROPOUT}"
    --attention-dropout "${ATTENTION_DROPOUT}"
    --impact-embedding-dim "${IMPACT_EMBED_DIM}"
    --batch-size "${BATCH_SIZE}"
    --epochs "${EPOCHS}"
    --val-size "${VAL_SIZE}"
    --seed "${SEED}"
    --grad-clip "${GRAD_CLIP}"
    --early-stop "${EARLY_STOP}"
    --num-workers "${NUM_WORKERS}"
    --use-layernorm "${USE_LAYERNORM}"
    --save-model "${SAVE_MODEL}"
    --output-root "${OUTPUT_ROOT}/${LABEL}"
)

echo "Job ID: ${SLURM_JOB_ID:-no_slurm_job_id}"
echo "Array task: ${task_id}/${n_tasks}"
echo "Label: ${LABEL}"
echo "Model type: ${MODEL_TYPE}"
echo "Trait: ${TRAIT}"
echo "Layers: ${LAYERS}"
echo "Epochs: ${EPOCHS}"
echo "Save model: ${SAVE_MODEL}"
echo "Train script: ${TRAIN_SCRIPT}"
echo "Python: $(command -v python3)"
echo "CUDA module=${CUDA_MODULE}"
echo "PyTorch module=${PYTORCH_MODULE}"
echo "Venv=${VENV_PATH}"
echo "Command: ${CMD[*]}"

module list 2>&1 || true
nvidia-smi || true

python3 - <<'PY'
import sys
import torch

print(f"python_executable={sys.executable}")
print(f"torch_file={torch.__file__}")
print(f"torch_version={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f"cuda_device_name={torch.cuda.get_device_name(0)}")
PY

"${CMD[@]}"
