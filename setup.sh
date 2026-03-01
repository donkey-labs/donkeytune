#!/bin/bash
set -euo pipefail

# =============================================================================
# setup.sh - Install dependencies for fine-tuning on Apple Silicon
# =============================================================================
# Installs: Python venv, mlx-lm (Apple MLX), llama.cpp (GGUF conversion)
# Usage: ./setup.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
LLAMA_CPP_DIR="${SCRIPT_DIR}/llama.cpp"

echo "============================================"
echo "  Fine-tuning Setup - Apple Silicon (MLX)"
echo "============================================"
echo ""

# --- Check: Apple Silicon ---
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "⚠  This script is designed for Apple Silicon (arm64)."
    echo "   Detected architecture: $(uname -m)"
    exit 1
fi

# --- Step 1: Create Python virtual environment ---
echo "→ Step 1/4: Python virtual environment..."

# Find Python 3.10+
PYTHON_BIN=""
for candidate in python3.13 python3.12 python3.11 python3.10; do
    if command -v "$candidate" &>/dev/null; then
        VERSION=$("$candidate" -c 'import sys; print(sys.version_info.minor)')
        MAJOR=$("$candidate" -c 'import sys; print(sys.version_info.major)')
        if [ "$MAJOR" -eq 3 ] && [ "$VERSION" -ge 10 ]; then
            PYTHON_BIN=$(command -v "$candidate")
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "Error: Python 3.10+ is required but not found."
    echo "Install it with: brew install python@3.12"
    exit 1
fi

echo "  Using: ${PYTHON_BIN} ($("$PYTHON_BIN" --version))"

# Delete existing venv if it was created with Python < 3.10
if [ -d "$VENV_DIR" ]; then
    VENV_PYTHON="${VENV_DIR}/bin/python3"
    VENV_MINOR=$("$VENV_PYTHON" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo "0")
    VENV_MAJOR=$("$VENV_PYTHON" -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo "0")
    if [ "$VENV_MAJOR" -lt 3 ] || { [ "$VENV_MAJOR" -eq 3 ] && [ "$VENV_MINOR" -lt 10 ]; }; then
        echo "  Existing venv uses Python ${VENV_MAJOR}.${VENV_MINOR} (< 3.10) — deleting it..."
        rm -rf "$VENV_DIR"
    else
        echo "  Existing venv OK (Python ${VENV_MAJOR}.${VENV_MINOR})"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    echo "  Created: ${VENV_DIR}"
fi

# Activate the venv
source "${VENV_DIR}/bin/activate"

# Load HuggingFace token if .env exists
if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
fi

# --- Step 2: Install mlx-lm ---
echo ""
echo "→ Step 2/4: Installing mlx-lm..."
pip install --upgrade pip > /dev/null 2>&1
pip install "mlx-lm[train]" 2>&1 | tail -1
echo "  mlx-lm installed ✓"

# --- Step 3: Install dependencies for GGUF conversion ---
echo ""
echo "→ Step 3/4: Cloning and building llama.cpp..."
if [ -d "$LLAMA_CPP_DIR" ]; then
    echo "  llama.cpp already exists, updating..."
    cd "$LLAMA_CPP_DIR"
    git pull --quiet
else
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
    cd "$LLAMA_CPP_DIR"
fi

# Build llama-quantize
echo "  Building llama-quantize..."
cmake -B build -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
cmake --build build --target llama-quantize -j "$(sysctl -n hw.ncpu)" 2>&1 | tail -1
echo "  llama-quantize built ✓"

# Install Python dependencies for convert_hf_to_gguf.py
cd "$SCRIPT_DIR"
pip install gguf numpy sentencepiece torch > /dev/null 2>&1
echo "  Conversion dependencies installed ✓"

# --- Step 4: Create working directories ---
echo ""
echo "→ Step 4/4: Creating directories..."
mkdir -p "${SCRIPT_DIR}/data"
mkdir -p "${SCRIPT_DIR}/output/adapters"
mkdir -p "${SCRIPT_DIR}/output/fused"
mkdir -p "${SCRIPT_DIR}/output/gguf"
echo "  Directories created ✓"

# --- Summary ---
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  Python venv  : ${VENV_DIR}"
echo "  mlx-lm       : $(pip show mlx-lm 2>/dev/null | grep Version | cut -d' ' -f2)"
echo "  llama.cpp     : ${LLAMA_CPP_DIR}"
echo "  llama-quantize: ${LLAMA_CPP_DIR}/build/bin/llama-quantize"
echo ""
echo "  To activate the venv:"
echo "    source ${VENV_DIR}/bin/activate"
echo ""
echo "  Next step: (for example)"
echo "    make train MODEL=unsloth/Qwen2.5-1.5B-Instruct"
echo ""
