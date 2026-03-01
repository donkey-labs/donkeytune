#!/bin/bash
set -euo pipefail

# =============================================================================
# test-model.sh - Test the fine-tuned model with Go prompts
# =============================================================================
# Usage:
#   ./test-model.sh                              (test with LoRA adapters)
#   ./test-model.sh --prompt "Write a Go HTTP server"
#   ./test-model.sh --fused                       (test with fused model)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Default parameters ---
MODEL="unsloth/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR="${SCRIPT_DIR}/output/adapters"
PROMPT=""
USE_FUSED=false
MAX_TOKENS=500

# --- Argument parsing ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)        MODEL="$2"; shift 2 ;;
        --adapter-path) ADAPTER_DIR="$2"; shift 2 ;;
        --prompt)       PROMPT="$2"; shift 2 ;;
        --fused)        USE_FUSED=true; shift ;;
        --max-tokens)   MAX_TOKENS="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL        Base model (default: $MODEL)"
            echo "  --adapter-path DIR   Adapter directory (default: ./output/adapters)"
            echo "  --prompt TEXT        Custom prompt"
            echo "  --fused              Use the fused model instead of adapters"
            echo "  --max-tokens N       Max tokens to generate (default: $MAX_TOKENS)"
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

# --- Default test prompts ---
DEFAULT_PROMPTS=(
    "Write a Go function that reverses a linked list"
    "Write a Go HTTP handler that accepts JSON and validates the input"
    "Write a Go function that reads a CSV file and returns a slice of structs"
)

echo "============================================"
echo "  Fine-tuned model test"
echo "============================================"
echo ""

if [ "$USE_FUSED" = true ]; then
    MODEL_ARG="--model ${SCRIPT_DIR}/output/fused"
    echo "  Mode: fused model"
else
    MODEL_ARG="--model ${MODEL} --adapter-path ${ADAPTER_DIR}"
    echo "  Mode: model + LoRA adapters"
fi
echo ""

# --- Run tests ---
if [ -n "${PROMPT}" ]; then
    # Single custom prompt
    echo "--- Prompt ---"
    echo "${PROMPT}"
    echo ""
    echo "--- Response ---"
    if [ "$USE_FUSED" = true ]; then
        mlx_lm.generate \
            --model "${SCRIPT_DIR}/output/fused" \
            --max-tokens "${MAX_TOKENS}" \
            --prompt "${PROMPT}"
    else
        mlx_lm.generate \
            --model "${MODEL}" \
            --adapter-path "${ADAPTER_DIR}" \
            --max-tokens "${MAX_TOKENS}" \
            --prompt "${PROMPT}"
    fi
else
    # Default test prompts
    for i in "${!DEFAULT_PROMPTS[@]}"; do
        echo "============================================"
        echo "  Test $((i+1))/${#DEFAULT_PROMPTS[@]}"
        echo "============================================"
        echo ""
        echo "--- Prompt ---"
        echo "${DEFAULT_PROMPTS[$i]}"
        echo ""
        echo "--- Response ---"
        if [ "$USE_FUSED" = true ]; then
            mlx_lm.generate \
                --model "${SCRIPT_DIR}/output/fused" \
                --max-tokens "${MAX_TOKENS}" \
                --prompt "${DEFAULT_PROMPTS[$i]}"
        else
            mlx_lm.generate \
                --model "${MODEL}" \
                --adapter-path "${ADAPTER_DIR}" \
                --max-tokens "${MAX_TOKENS}" \
                --prompt "${DEFAULT_PROMPTS[$i]}"
        fi
        echo ""
        echo ""
    done
fi

echo "============================================"
echo "  Tests complete"
echo "============================================"
