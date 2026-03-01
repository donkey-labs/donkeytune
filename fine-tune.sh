#!/bin/bash
set -euo pipefail

# =============================================================================
# fine-tune.sh - LoRA fine-tuning with mlx-lm
# =============================================================================
# Usage:
#   ./fine-tune.sh                                    (default parameters)
#   ./fine-tune.sh --model unsloth/Qwen2.5-0.5B-Instruct
#   ./fine-tune.sh --iters 500 --batch-size 2
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Default parameters ---
MODEL="unsloth/Qwen2.5-1.5B-Instruct"
DATA_DIR="${SCRIPT_DIR}/data"
ADAPTER_DIR="${SCRIPT_DIR}/output/adapters"
BATCH_SIZE=4
NUM_LAYERS=8
ITERS=600
LEARNING_RATE="1e-5"
LORA_RANK=8
RESUME_ADAPTER=""
STEPS_PER_REPORT=10
STEPS_PER_EVAL=50
VAL_BATCHES=5
SAVE_EVERY=100

# --- Argument parsing ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)         MODEL="$2"; shift 2 ;;
        --data)          DATA_DIR="$2"; shift 2 ;;
        --adapter-path)  ADAPTER_DIR="$2"; shift 2 ;;
        --batch-size)    BATCH_SIZE="$2"; shift 2 ;;
        --num-layers)    NUM_LAYERS="$2"; shift 2 ;;
        --iters)         ITERS="$2"; shift 2 ;;
        --lr)            LEARNING_RATE="$2"; shift 2 ;;
        --rank)          LORA_RANK="$2"; shift 2 ;;
        --resume)        RESUME_ADAPTER="$2"; shift 2 ;;
        --save-every)    SAVE_EVERY="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL        HuggingFace model (default: $MODEL)"
            echo "  --data DIR           Data directory (default: ./data)"
            echo "  --adapter-path DIR   Adapter output directory (default: ./output/adapters)"
            echo "  --batch-size N       Batch size (default: $BATCH_SIZE)"
            echo "  --num-layers N       Number of LoRA layers (default: $NUM_LAYERS)"
            echo "  --iters N            Number of iterations (default: $ITERS)"
            echo "  --lr RATE            Learning rate (default: $LEARNING_RATE)"
            echo "  --rank N             LoRA rank (default: $LORA_RANK)"
            echo "  --resume PATH        Resume from existing adapters file"
            echo "  --save-every N       Save every N steps (default: $SAVE_EVERY)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Check venv ---
if [ ! -d "${SCRIPT_DIR}/.venv" ]; then
    echo "Error: Venv not found. Run first: ./setup.sh"
    exit 1
fi
source "${SCRIPT_DIR}/.venv/bin/activate"

# Load HuggingFace token if .env exists
if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
fi

# --- Check data ---
if [ ! -f "${DATA_DIR}/train.jsonl" ]; then
    echo "Error: ${DATA_DIR}/train.jsonl not found."
    exit 1
fi

TRAIN_COUNT=$(wc -l < "${DATA_DIR}/train.jsonl" | tr -d ' ')
VALID_COUNT=0
if [ -f "${DATA_DIR}/valid.jsonl" ]; then
    VALID_COUNT=$(wc -l < "${DATA_DIR}/valid.jsonl" | tr -d ' ')
fi

# --- Summary ---
echo "============================================"
echo "  Fine-tuning LoRA - mlx-lm"
echo "============================================"
echo ""
echo "  Model        : ${MODEL}"
echo "  Data          : ${DATA_DIR}"
echo "  Train         : ${TRAIN_COUNT} examples"
echo "  Validation    : ${VALID_COUNT} examples"
echo "  Adapters      : ${ADAPTER_DIR}"
echo ""
if [ -n "${RESUME_ADAPTER}" ]; then
echo "  Resume from   : ${RESUME_ADAPTER}"
fi
echo ""
echo "  Hyperparameters:"
echo "    Batch size   : ${BATCH_SIZE}"
echo "    Num layers   : ${NUM_LAYERS}"
echo "    Iterations   : ${ITERS}"
echo "    Learning rate: ${LEARNING_RATE}"
echo "    LoRA rank    : ${LORA_RANK}"
echo ""

# --- Create output directory ---
mkdir -p "${ADAPTER_DIR}"

# --- Generate YAML config file (for LoRA rank) ---
CONFIG_FILE="${ADAPTER_DIR}/train_config.yaml"
cat > "${CONFIG_FILE}" <<EOF
lora_parameters:
  rank: ${LORA_RANK}
  alpha: $(( LORA_RANK * 2 ))
  dropout: 0.05
  scale: 10.0
EOF

# --- Start fine-tuning ---
echo "→ Starting fine-tuning..."
echo ""

mlx_lm.lora \
    --model "${MODEL}" \
    --train \
    --data "${DATA_DIR}" \
    --adapter-path "${ADAPTER_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --num-layers "${NUM_LAYERS}" \
    --iters "${ITERS}" \
    --learning-rate "${LEARNING_RATE}" \
    --steps-per-report "${STEPS_PER_REPORT}" \
    --steps-per-eval "${STEPS_PER_EVAL}" \
    --val-batches "${VAL_BATCHES}" \
    --save-every "${SAVE_EVERY}" \
    -c "${CONFIG_FILE}" \
    ${RESUME_ADAPTER:+--resume-adapter-file "${RESUME_ADAPTER}"}

echo ""
echo "============================================"
echo "  Fine-tuning complete!"
echo "============================================"
echo "  Adapters saved to: ${ADAPTER_DIR}"
echo ""
echo "  Next step:"
echo "    ./convert-to-gguf.sh --model ${MODEL}"
echo ""
