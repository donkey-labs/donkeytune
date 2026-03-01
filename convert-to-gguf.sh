#!/bin/bash
set -euo pipefail

# =============================================================================
# convert-to-gguf.sh - Convert the fine-tuned model to GGUF
# =============================================================================
# Pipeline: fuse adapters → convert to GGUF → quantize
# Usage:
#   ./convert-to-gguf.sh
#   ./convert-to-gguf.sh --model unsloth/Qwen2.5-0.5B-Instruct --quantize Q5_K_M
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Default parameters ---
MODEL="unsloth/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR="${SCRIPT_DIR}/output/adapters"
FUSED_DIR="${SCRIPT_DIR}/output/fused"
GGUF_DIR="${SCRIPT_DIR}/output/gguf"
QUANTIZE_TYPE="Q4_K_M"
OUTPUT_NAME=""

# --- Argument parsing ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)        MODEL="$2"; shift 2 ;;
        --adapter-path) ADAPTER_DIR="$2"; shift 2 ;;
        --output-dir)   GGUF_DIR="$2"; shift 2 ;;
        --quantize)     QUANTIZE_TYPE="$2"; shift 2 ;;
        --name)         OUTPUT_NAME="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL        Base HuggingFace model (default: $MODEL)"
            echo "  --adapter-path DIR   LoRA adapter directory (default: ./output/adapters)"
            echo "  --output-dir DIR     GGUF output directory (default: ./output/gguf)"
            echo "  --quantize TYPE      Quantization type (default: Q4_K_M)"
            echo "  --name NAME          Output file name (default: auto)"
            echo ""
            echo "Common quantization types:"
            echo "  Q4_K_M   4-bit (recommended, good size/quality tradeoff)"
            echo "  Q5_K_M   5-bit (better quality, slightly larger)"
            echo "  Q8_0     8-bit (near full precision)"
            echo "  f16      16-bit (full precision)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Auto output name ---
if [ -z "${OUTPUT_NAME}" ]; then
    # Extract model name: "Qwen/Qwen2.5-Coder-0.5B-Instruct" → "Qwen2.5-Coder-0.5B-Instruct"
    BASE_NAME=$(basename "${MODEL}")
    OUTPUT_NAME="${BASE_NAME}-finetuned"
fi

# --- Checks ---
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

if [ ! -d "${ADAPTER_DIR}" ] || [ ! -f "${ADAPTER_DIR}/adapters.safetensors" ]; then
    echo "Error: Adapters not found in ${ADAPTER_DIR}"
    echo "Run first: ./fine-tune.sh"
    exit 1
fi

LLAMA_CPP="${SCRIPT_DIR}/llama.cpp"
if [ ! -d "${LLAMA_CPP}" ]; then
    echo "Error: llama.cpp not found. Run first: ./setup.sh"
    exit 1
fi

echo "============================================"
echo "  GGUF Conversion"
echo "============================================"
echo ""
echo "  Model        : ${MODEL}"
echo "  Adapters      : ${ADAPTER_DIR}"
echo "  Quantization  : ${QUANTIZE_TYPE}"
echo "  Output        : ${GGUF_DIR}/${OUTPUT_NAME}-${QUANTIZE_TYPE}.gguf"
echo ""

mkdir -p "${FUSED_DIR}" "${GGUF_DIR}"

# --- Step 1: Fuse LoRA adapters into the model ---
echo "→ Step 1/3: Fusing LoRA adapters..."
mlx_lm.fuse \
    --model "${MODEL}" \
    --adapter-path "${ADAPTER_DIR}" \
    --save-path "${FUSED_DIR}" \
    --dequantize

echo "  Fused model saved to: ${FUSED_DIR} ✓"

# --- Post-fuse fixes ---
# mlx_lm.fuse may not copy all tokenizer files needed by convert_hf_to_gguf.py.
# Download missing tokenizer files (e.g. tokenizer.model) from the base model.
python3 -c "
import json, os
from huggingface_hub import hf_hub_download, list_repo_files

fused_dir = '${FUSED_DIR}'
model_id = '${MODEL}'

# 1) Copy missing tokenizer files from the base model
#    (convert_hf_to_gguf.py needs tokenizer.model for SentencePiece models)
needed_files = ['tokenizer.model', 'added_tokens.json', 'special_tokens_map.json']
try:
    repo_files = set(list_repo_files(model_id))
    for fname in needed_files:
        dest = os.path.join(fused_dir, fname)
        if not os.path.exists(dest) and fname in repo_files:
            hf_hub_download(model_id, fname, local_dir=fused_dir)
            print(f'  Copied {fname} from base model')
except Exception as e:
    print(f'  Warning: could not fetch tokenizer files: {e}')

# 2) Fix vocab_size if tokenizer has more tokens than config declares
#    Some models (e.g. Gemma 3) have extra tokens that exceed vocab_size
config_path = os.path.join(fused_dir, 'config.json')
with open(config_path) as f:
    config = json.load(f)

vocab_size = config.get('vocab_size', 0)

tok_path = os.path.join(fused_dir, 'tokenizer.json')
if os.path.exists(tok_path):
    with open(tok_path) as f:
        tok = json.load(f)
    max_id = 0
    if 'model' in tok and 'vocab' in tok['model']:
        max_id = len(tok['model']['vocab']) - 1
    if 'added_tokens' in tok:
        for t in tok['added_tokens']:
            if t.get('id', 0) > max_id:
                max_id = t['id']
    actual_size = max_id + 1
    if actual_size > vocab_size:
        config['vocab_size'] = actual_size
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f'  Fixed vocab_size: {vocab_size} → {actual_size}')
    else:
        print(f'  Vocab size OK: {vocab_size}')
"

# --- Step 2: Convert to GGUF (f16) ---
echo ""
echo "→ Step 2/3: Converting to GGUF (f16)..."
F16_GGUF="${GGUF_DIR}/${OUTPUT_NAME}-f16.gguf"

python3 "${LLAMA_CPP}/convert_hf_to_gguf.py" \
    "${FUSED_DIR}" \
    --outfile "${F16_GGUF}" \
    --outtype f16

echo "  GGUF f16 created: ${F16_GGUF} ✓"

# --- Step 3: Quantize ---
echo ""
echo "→ Step 3/3: Quantizing ${QUANTIZE_TYPE}..."
QUANTIZED_GGUF="${GGUF_DIR}/${OUTPUT_NAME}-${QUANTIZE_TYPE}.gguf"

QUANTIZE_BIN="${LLAMA_CPP}/build/bin/llama-quantize"
if [ ! -f "${QUANTIZE_BIN}" ]; then
    echo "Error: llama-quantize not found. Rebuild with ./setup.sh"
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
