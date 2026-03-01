# =============================================================================
# Makefile - Fine-tuning pipeline for small models on Apple Silicon
# =============================================================================
#
# Quick usage:
#   make setup                    # Install dependencies (once)
#   make train                    # Fine-tune the model
#   make convert                  # Convert to GGUF
#   make test                     # Test the model
#   make all                      # Run the full pipeline at once
#
# Customize:
#   make train MODEL=meta-llama/Llama-3.2-1B-Instruct ITERS=1000
#   make convert QUANTIZE=Q5_K_M
#
# =============================================================================

# --- Configurable variables ---
MODEL       ?= unsloth/Qwen2.5-1.5B-Instruct
DATA_DIR    ?= ./data
BATCH_SIZE  ?= 4
NUM_LAYERS  ?= 8
ITERS       ?= 600
LR          ?= 1e-5
RANK        ?= 8
QUANTIZE    ?= Q4_K_M
OUTPUT_NAME ?=
INPUTS      ?=
RESUME_FROM ?=

.PHONY: help setup train convert hf-to-gguf test test-prompt all clean clean-output \
        merge-data train-resume train-merged

help: ## Show this help
	@echo ""
	@echo "  Fine-tuning Pipeline - Apple Silicon (MLX)"
	@echo "  ==========================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Variables:"
	@echo "    MODEL=$(MODEL)"
	@echo "    ITERS=$(ITERS)  BATCH_SIZE=$(BATCH_SIZE)  LR=$(LR)  RANK=$(RANK)"
	@echo "    QUANTIZE=$(QUANTIZE)"
	@echo ""

setup: ## Install dependencies (venv, mlx-lm, llama.cpp)
	@./setup.sh

train: ## Run LoRA fine-tuning
	@./fine-tune.sh \
		--model $(MODEL) \
		--data $(DATA_DIR) \
		--batch-size $(BATCH_SIZE) \
		--num-layers $(NUM_LAYERS) \
		--iters $(ITERS) \
		--lr $(LR) \
		--rank $(RANK)

convert: ## Convert the fine-tuned model to GGUF
	@./convert-to-gguf.sh \
		--model $(MODEL) \
		--quantize $(QUANTIZE) \
		$(if $(OUTPUT_NAME),--name $(OUTPUT_NAME),)

hf-to-gguf: ## Convert a HuggingFace model to GGUF (no fine-tuning)
	@./hf-to-gguf.sh \
		--model $(MODEL) \
		--quantize $(QUANTIZE) \
		$(if $(OUTPUT_NAME),--name $(OUTPUT_NAME),)

test: ## Test the fine-tuned model with Go prompts
	@./test-model.sh --model $(MODEL)

test-prompt: ## Test with a custom prompt (PROMPT="...")
	@./test-model.sh --model $(MODEL) --prompt "$(PROMPT)"

all: train convert test ## Full pipeline: train → convert → test

clean-output: ## Delete output files (adapters, fused, gguf)
	@echo "Deleting output files..."
	@rm -rf output/adapters/* output/fused/* output/gguf/*
	@echo "Cleaned ✓"

clean: clean-output ## Clean everything (output + venv + llama.cpp)
	@echo "Deleting venv and llama.cpp..."
	@rm -rf .venv llama.cpp
	@echo "Cleaned ✓"

# =============================================================================
# Multi-dataset targets
# =============================================================================

merge-data: ## Merge datasets: INPUTS="./datasets/a ./datasets/b" OUTPUT=./data
	@./merge-datasets.sh --inputs $(INPUTS) --output $(DATA_DIR)

train-resume: ## Resume training on new data with existing adapters
	@./fine-tune.sh \
		--model $(MODEL) \
		--data $(DATA_DIR) \
		--batch-size $(BATCH_SIZE) \
		--num-layers $(NUM_LAYERS) \
		--iters $(ITERS) \
		--lr $(LR) \
		--rank $(RANK) \
		--resume $(RESUME_FROM)

train-merged: ## Merge datasets then resume training with existing adapters
	@./merge-datasets.sh --inputs $(INPUTS) --output ./output/merged-data
	@./fine-tune.sh \
		--model $(MODEL) \
		--data ./output/merged-data \
		--batch-size $(BATCH_SIZE) \
		--num-layers $(NUM_LAYERS) \
		--iters $(ITERS) \
		--lr $(LR) \
		--rank $(RANK) \
		--resume $(RESUME_FROM)
