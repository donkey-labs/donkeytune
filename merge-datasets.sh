#!/bin/bash
set -euo pipefail

# =============================================================================
# merge-datasets.sh - Merge multiple datasets into one
# =============================================================================
# Concatenates train.jsonl and valid.jsonl from multiple dataset directories.
# Usage:
#   ./merge-datasets.sh --inputs ./datasets/goloscript ./datasets/hawaiian-pizza --output ./data
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Default parameters ---
INPUTS=()
OUTPUT_DIR=""

# --- Argument parsing ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --inputs)
            shift
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                INPUTS+=("$1")
                shift
            done
            ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 --inputs DIR1 DIR2 [DIR3 ...] --output DIR"
            echo ""
            echo "Options:"
            echo "  --inputs DIR ...   Dataset directories to merge (each must contain train.jsonl)"
            echo "  --output DIR       Output directory for merged files"
            echo ""
            echo "Example:"
            echo "  $0 --inputs ./datasets/goloscript ./datasets/hawaiian-pizza --output ./data"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Validation ---
if [ ${#INPUTS[@]} -lt 2 ]; then
    echo "Error: At least 2 input directories required."
    echo "Usage: $0 --inputs DIR1 DIR2 --output DIR"
    exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
    echo "Error: --output is required."
    exit 1
fi

for dir in "${INPUTS[@]}"; do
    if [ ! -f "${dir}/train.jsonl" ]; then
        echo "Error: ${dir}/train.jsonl not found."
        exit 1
    fi
done

# --- Merge ---
mkdir -p "${OUTPUT_DIR}"

# Clear output files
> "${OUTPUT_DIR}/train.jsonl"
> "${OUTPUT_DIR}/valid.jsonl"

TRAIN_TOTAL=0
VALID_TOTAL=0

echo "============================================"
echo "  Merging datasets"
echo "============================================"
echo ""

for dir in "${INPUTS[@]}"; do
    NAME=$(basename "${dir}")
    TRAIN_COUNT=$(wc -l < "${dir}/train.jsonl" | tr -d ' ')
    TRAIN_TOTAL=$((TRAIN_TOTAL + TRAIN_COUNT))

    cat "${dir}/train.jsonl" >> "${OUTPUT_DIR}/train.jsonl"

    if [ -f "${dir}/valid.jsonl" ]; then
        VALID_COUNT=$(wc -l < "${dir}/valid.jsonl" | tr -d ' ')
        VALID_TOTAL=$((VALID_TOTAL + VALID_COUNT))
        cat "${dir}/valid.jsonl" >> "${OUTPUT_DIR}/valid.jsonl"
    fi

    echo "  + ${NAME}: ${TRAIN_COUNT} train examples"
done

echo ""
echo "  Output    : ${OUTPUT_DIR}"
echo "  Total     : ${TRAIN_TOTAL} train, ${VALID_TOTAL} valid"
echo ""
echo "  Done ✓"
echo ""
