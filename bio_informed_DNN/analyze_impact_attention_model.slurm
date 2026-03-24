#!/bin/bash
#SBATCH --job-name=impact_attn_interp
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/ulg/genan/jrfelix/projects/Snp_models/bio_informed_DNN/logs/impact_attention_interp_%j.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/ulg/genan/jrfelix/projects/Snp_models}
SCRIPT_DIR=${SCRIPT_DIR:-${PROJECT_ROOT}/bio_informed_DNN}
VENV_PATH=${VENV_PATH:-${PROJECT_ROOT}/venv311}

cd "${SCRIPT_DIR}"
mkdir -p logs

module load CUDA
source "${VENV_PATH}/bin/activate"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export DISABLE_PANDERA_IMPORT_WARNING=True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CHECKPOINT=${CHECKPOINT:-./saved_models/vep_ingenes_newimpact/critHuberLoss_gelu/impact_attention/model_vep_ingenes_newimpact_impact_attention_layers50_emb16_attndrop0.05_lr0.0001_trait2_epoch150_critHuberLoss_actgelu_batch128_wdecay0.0001_dropout0.5_gclip1.0_seed43_eaTrue_2026-03-23_16-58-13.pt}
OUTPUT_DIR=${OUTPUT_DIR:-}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-128}
TOP_SNPS=${TOP_SNPS:-30}
TOP_GENES=${TOP_GENES:-20}
TOP_IMPACT_SIZES=${TOP_IMPACT_SIZES:-"50 100 200"}
GENE_GLOBAL_TOP_K=${GENE_GLOBAL_TOP_K:-0}
CORRELATION_BLOCK_TOP_K=${CORRELATION_BLOCK_TOP_K:-0}
SNP_GLOBAL_TOP_K=${SNP_GLOBAL_TOP_K:-0}
PERMUTATION_REPEATS=${PERMUTATION_REPEATS:-3}
CORRELATION_THRESHOLD=${CORRELATION_THRESHOLD:-0.60}
CORRELATION_MAX_GAP_KB=${CORRELATION_MAX_GAP_KB:-500}
IG_STEPS=${IG_STEPS:-32}
IG_SAMPLES=${IG_SAMPLES:-128}
MAX_SAMPLES=${MAX_SAMPLES:-}
RUN_SHAP=${RUN_SHAP:-false}
SHAP_BACKGROUND_SIZE=${SHAP_BACKGROUND_SIZE:-64}
SHAP_EVAL_SIZE=${SHAP_EVAL_SIZE:-64}
EVAL_SAMPLE_ID_FILE=${EVAL_SAMPLE_ID_FILE:-}
EVAL_SAMPLE_ID_COLUMN=${EVAL_SAMPLE_ID_COLUMN:-}

CMD=(
    python3 analyze_impact_attention_model.py
    --checkpoint "${CHECKPOINT}"
    --device "${DEVICE}"
    --batch-size "${BATCH_SIZE}"
    --top-snps "${TOP_SNPS}"
    --top-genes "${TOP_GENES}"
    --gene-global-top-k "${GENE_GLOBAL_TOP_K}"
    --correlation-block-top-k "${CORRELATION_BLOCK_TOP_K}"
    --snp-global-top-k "${SNP_GLOBAL_TOP_K}"
    --permutation-repeats "${PERMUTATION_REPEATS}"
    --correlation-threshold "${CORRELATION_THRESHOLD}"
    --correlation-max-gap-kb "${CORRELATION_MAX_GAP_KB}"
    --ig-steps "${IG_STEPS}"
    --ig-samples "${IG_SAMPLES}"
    --shap-background-size "${SHAP_BACKGROUND_SIZE}"
    --shap-eval-size "${SHAP_EVAL_SIZE}"
    --top-impact-sizes
)

read -r -a TOP_IMPACT_ARRAY <<< "${TOP_IMPACT_SIZES}"
CMD+=("${TOP_IMPACT_ARRAY[@]}")

if [[ -n "${OUTPUT_DIR}" ]]; then
    CMD+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
    CMD+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ "${RUN_SHAP,,}" == "true" ]]; then
    CMD+=(--run-shap)
fi

if [[ -n "${EVAL_SAMPLE_ID_FILE}" ]]; then
    CMD+=(--sample-id-file "${EVAL_SAMPLE_ID_FILE}")
fi

if [[ -n "${EVAL_SAMPLE_ID_COLUMN}" ]]; then
    CMD+=(--sample-id-column "${EVAL_SAMPLE_ID_COLUMN}")
fi

echo "Job ID: ${SLURM_JOB_ID:-no_slurm_job_id}"
echo "Working directory: ${SCRIPT_DIR}"
echo "Python: $(command -v python3)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Device: ${DEVICE}"
echo "Run SHAP: ${RUN_SHAP}"
echo "Command: ${CMD[*]}"

"${CMD[@]}"
