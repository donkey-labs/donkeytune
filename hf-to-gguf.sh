#!/bin/bash
set -euo pipefail

# =============================================================================
# hf-to-gguf.sh - Download a HuggingFace model and convert it to GGUF
# =============================================================================
# Pipeline: download model → convert to GGUF (f16) → quantize
# Usage:
#   ./hf-to-gguf.sh --model Nanbeige/Nanbeige4.1-3B
#   ./hf-to-gguf.sh --model unsloth/gemma-3-270m-it --quantize Q5_K_M
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Default parameters ---
MODEL=""
GGUF_DIR="${SCRIPT_DIR}/output/gguf"
QUANTIZE_TYPE="Q4_K_M"
OUTPUT_NAME=""

# --- Argument parsing ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)        MODEL="$2"; shift 2 ;;
        --output-dir)   GGUF_DIR="$2"; shift 2 ;;
        --quantize)     QUANTIZE_TYPE="$2"; shift 2 ;;
        --name)         OUTPUT_NAME="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL        HuggingFace model ID (required)"
            echo "  --output-dir DIR     GGUF output directory (default: ./output/gguf)"
            echo "  --quantize TYPE      Quantization type (default: Q4_K_M)"
            echo "  --name NAME          Output file name (default: auto)"
            echo ""
            echo "Common quantization types:"
            echo "  Q4_K_M   4-bit (recommended, good size/quality tradeoff)"
            echo "  Q5_K_M   5-bit (better quality, slightly larger)"
            echo "  Q8_0     8-bit (near full precision)"
            echo "  f16      16-bit (full precision)"
            echo ""
            echo "Examples:"
            echo "  $0 --model Nanbeige/Nanbeige4.1-3B"
            echo "  $0 --model unsloth/gemma-3-270m-it --quantize Q8_0"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "${MODEL}" ]; then
    echo "Error: --model is required"
    echo "Usage: $0 --model <huggingface-model-id>"
    exit 1
fi

# --- Auto output name ---
if [ -z "${OUTPUT_NAME}" ]; then
    BASE_NAME=$(basename "${MODEL}")
    OUTPUT_NAME="${BASE_NAME}"
fi

# --- Checks ---
if [ ! -d "${SCRIPT_DIR}/.venv" ]; then
    echo "Error: Venv not found. Run first: make setup"
    exit 1
fi
source "${SCRIPT_DIR}/.venv/bin/activate"

# Load HuggingFace token if .env exists
if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
fi

LLAMA_CPP="${SCRIPT_DIR}/llama.cpp"
if [ ! -d "${LLAMA_CPP}" ]; then
    echo "Error: llama.cpp not found. Run first: make setup"
    exit 1
fi

echo "============================================"
echo "  HuggingFace → GGUF Conversion"
echo "============================================"
echo ""
echo "  Model        : ${MODEL}"
echo "  Quantization  : ${QUANTIZE_TYPE}"
echo "  Output        : ${GGUF_DIR}/${OUTPUT_NAME}-${QUANTIZE_TYPE}.gguf"
echo ""

mkdir -p "${GGUF_DIR}"

# --- Step 1: Download the model from HuggingFace ---
echo "→ Step 1/3: Downloading model from HuggingFace..."
MODEL_DIR=$(python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('${MODEL}')
print(path)
")
echo "  Model downloaded to: ${MODEL_DIR} ✓"

# --- Step 2: Convert to GGUF (f16) ---
echo ""
echo "→ Step 2/3: Converting to GGUF (f16)..."
F16_GGUF="${GGUF_DIR}/${OUTPUT_NAME}-f16.gguf"

python3 "${LLAMA_CPP}/convert_hf_to_gguf.py" \
    "${MODEL_DIR}" \
    --outfile "${F16_GGUF}" \
    --outtype f16

echo "  GGUF f16 created: ${F16_GGUF} ✓"

# --- Step 3: Quantize ---
echo ""
echo "→ Step 3/3: Quantizing ${QUANTIZE_TYPE}..."
QUANTIZED_GGUF="${GGUF_DIR}/${OUTPUT_NAME}-${QUANTIZE_TYPE}.gguf"

QUANTIZE_BIN="${LLAMA_CPP}/build/bin/llama-quantize"
if [ ! -f "${QUANTIZE_BIN}" ]; then
    echo "Error: llama-quantize not found. Rebuild with: make setup"
    exit 1
fi

"${QUANTIZE_BIN}" "${F16_GGUF}" "${QUANTIZED_GGUF}" "${QUANTIZE_TYPE}"

echo ""
echo "============================================"
echo "  Conversion complete!"
echo "============================================"
echo ""
echo "  Output files:"
echo "    f16           : ${F16_GGUF}"
echo "    ${QUANTIZE_TYPE}       : ${QUANTIZED_GGUF}"
echo ""

# File sizes
F16_SIZE=$(du -h "${F16_GGUF}" | cut -f1)
Q_SIZE=$(du -h "${QUANTIZED_GGUF}" | cut -f1)
echo "  Sizes:"
echo "    f16           : ${F16_SIZE}"
echo "    ${QUANTIZE_TYPE}       : ${Q_SIZE}"
echo ""

