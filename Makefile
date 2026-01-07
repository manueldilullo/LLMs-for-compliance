# Makefile for LLMs-for-compliance Project
# Manages dependencies, GPU setup, and common tasks

.PHONY: help install install-cpu install-gpu install-dev install-optional \
        check-gpu check-cuda setup-cuda install-llama-gpu install-llama-cpu \
        install-datapizza install-unsloth test clean

# Default target
.DEFAULT_GOAL := help

# Python interpreter
PYTHON := python
PIP := $(PYTHON) -m pip

# CUDA detection
CUDA_VERSION := $(shell nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//' || echo "not-found")
HAS_CUDA := $(shell command -v nvcc 2>/dev/null && echo "yes" || echo "no")

# Color output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

##@ Help

help: ## Display this help message
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  LLMs-for-compliance - Project Setup$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(CYAN)<target>$(NC)\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(CYAN)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

##@ Installation

install: check-python ## Install all dependencies (auto-detect GPU)
	@echo "$(GREEN)Installing base dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@if [ "$(HAS_CUDA)" = "yes" ]; then \
		echo "$(GREEN)GPU detected! Installing GPU-specific packages...$(NC)"; \
		$(MAKE) install-gpu; \
	else \
		echo "$(YELLOW)No GPU detected. Installing CPU-only packages...$(NC)"; \
		$(MAKE) install-cpu; \
	fi
	@echo "$(GREEN)✓ Installation complete!$(NC)"

install-cpu: ## Install CPU-only dependencies
	@echo "$(BLUE)Installing CPU-only packages...$(NC)"
	$(PIP) install faiss-cpu
	$(PIP) install llama-cpp-python
	@echo "$(GREEN)✓ CPU packages installed$(NC)"

install-gpu: check-cuda ## Install GPU-accelerated dependencies
	@echo "$(BLUE)Installing GPU-accelerated packages...$(NC)"
	@echo "$(YELLOW)CUDA Version: $(CUDA_VERSION)$(NC)"
	@if [ "$(CUDA_VERSION)" = "not-found" ]; then \
		echo "$(RED)ERROR: CUDA not found. Install CUDA first or use 'make install-cpu'$(NC)"; \
		exit 1; \
	fi
	$(MAKE) install-faiss-gpu
	$(MAKE) install-llama-gpu
	@echo "$(GREEN)✓ GPU packages installed$(NC)"

install-faiss-gpu: ## Install FAISS with GPU support
	@echo "$(BLUE)Installing FAISS-GPU...$(NC)"
	@if echo "$(CUDA_VERSION)" | grep -q "^12"; then \
		$(PIP) install faiss-gpu-cu12; \
	elif echo "$(CUDA_VERSION)" | grep -q "^11"; then \
		$(PIP) install faiss-gpu; \
	else \
		echo "$(YELLOW)Unknown CUDA version, installing default faiss-gpu$(NC)"; \
		$(PIP) install faiss-gpu; \
	fi

install-llama-gpu: ## Install llama-cpp-python with GPU support
	@echo "$(BLUE)Installing llama-cpp-python with CUDA support...$(NC)"
	@echo "$(YELLOW)This may take several minutes...$(NC)"
	CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native" \
	FORCE_CMAKE=1 \
	$(PIP) install llama-cpp-python --force-reinstall --no-cache-dir --verbose

install-llama-cpu: ## Install llama-cpp-python CPU-only
	@echo "$(BLUE)Installing llama-cpp-python (CPU)...$(NC)"
	$(PIP) install llama-cpp-python

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install jupyter ipykernel matplotlib seaborn pytest black flake8 mypy
	@echo "$(GREEN)✓ Development tools installed$(NC)"

install-optional: ## Install optional dependencies (Datapizza, Unsloth)
	@echo "$(BLUE)Installing optional dependencies...$(NC)"
	$(MAKE) install-datapizza
	$(MAKE) install-unsloth
	@echo "$(GREEN)✓ Optional packages installed$(NC)"

install-datapizza: ## Install Datapizza AI for Qdrant RAG
	@echo "$(BLUE)Installing Datapizza AI...$(NC)"
	$(PIP) install datapizza-ai datapizza-ai-parsers-docling datapizza-ai-embedders-fastembedder
	$(PIP) install qdrant-client
	@echo "$(GREEN)✓ Datapizza installed$(NC)"

install-unsloth: ## Install Unsloth for efficient fine-tuning
	@echo "$(BLUE)Installing Unsloth...$(NC)"
	$(PIP) install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
	@echo "$(GREEN)✓ Unsloth installed$(NC)"

##@ System Checks

check-python: ## Check Python version
	@echo "$(BLUE)Checking Python version...$(NC)"
	@$(PYTHON) --version
	@$(PYTHON) -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" || \
		(echo "$(RED)ERROR: Python 3.8+ required$(NC)" && exit 1)
	@echo "$(GREEN)✓ Python version OK$(NC)"

check-cuda: ## Check CUDA availability
	@echo "$(BLUE)Checking CUDA installation...$(NC)"
	@if [ "$(HAS_CUDA)" = "yes" ]; then \
		echo "$(GREEN)✓ CUDA found: $(CUDA_VERSION)$(NC)"; \
		nvcc --version; \
	else \
		echo "$(YELLOW)⚠ CUDA not found$(NC)"; \
		echo "Install CUDA from: https://developer.nvidia.com/cuda-downloads"; \
	fi

check-gpu: ## Check GPU availability and details
	@echo "$(BLUE)GPU Information:$(NC)"
	@$(PYTHON) -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU devices:', torch.cuda.device_count() if torch.cuda.is_available() else 0); [print(f'  - Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None" 2>/dev/null || echo "$(YELLOW)PyTorch not installed yet$(NC)"
	@which nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || echo "$(YELLOW)nvidia-smi not available$(NC)"

setup-cuda: ## Setup CUDA environment variables
	@echo "$(BLUE)Setting up CUDA environment...$(NC)"
	@echo "export CUDA_HOME=/usr/local/cuda"
	@echo "export PATH=\$$CUDA_HOME/bin:\$$PATH"
	@echo "export LD_LIBRARY_PATH=\$$CUDA_HOME/lib64:\$$LD_LIBRARY_PATH"
	@echo ""
	@echo "$(YELLOW)Add these to your ~/.bashrc or ~/.zshrc$(NC)"

##@ Testing

test: ## Run basic import tests
	@echo "$(BLUE)Running import tests...$(NC)"
	@$(PYTHON) -c "import torch; print('✓ PyTorch:', torch.__version__)"
	@$(PYTHON) -c "import transformers; print('✓ Transformers:', transformers.__version__)"
	@$(PYTHON) -c "import sentence_transformers; print('✓ SentenceTransformers')"
	@$(PYTHON) -c "import faiss; print('✓ FAISS')"
	@$(PYTHON) -c "import llama_cpp; print('✓ llama-cpp-python')"
	@$(PYTHON) -c "import networkx; print('✓ NetworkX')"
	@$(PYTHON) -c "import pydantic; print('✓ Pydantic')"
	@echo "$(GREEN)✓ All core imports successful$(NC)"

test-gpu: ## Test GPU functionality
	@echo "$(BLUE)Testing GPU functionality...$(NC)"
	@$(PYTHON) -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('✓ PyTorch CUDA available'); print('  Device:', torch.cuda.get_device_name(0))"
	@$(PYTHON) -c "import faiss; print('✓ FAISS version:', faiss.__version__); res = faiss.StandardGpuResources(); print('  GPU resources initialized')" 2>/dev/null || echo "$(YELLOW)⚠ FAISS GPU not available$(NC)"
	@echo "$(GREEN)✓ GPU tests complete$(NC)"

test-modules: ## Test custom module imports
	@echo "$(BLUE)Testing custom modules...$(NC)"
	@$(PYTHON) -c "from retrieval_augmented_generation import RAGFactory; print('✓ retrieval_augmented_generation')"
	@$(PYTHON) -c "from finetuning import load_model; print('✓ finetuning')"
	@$(PYTHON) -c "from synthetic_dataset_creation import AsyncQAGDPRPipeline; print('✓ synthetic_dataset_creation')"
	@$(PYTHON) test_init_llm.py
	@$(PYTHON) test_datapizza_rag.py
	@echo "$(GREEN)✓ All modules working$(NC)"

##@ Cleanup

clean: ## Clean up cache and temporary files
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-models: ## Remove downloaded models (use with caution!)
	@echo "$(RED)WARNING: This will delete all downloaded models$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf ./models/*; \
		echo "$(GREEN)Models deleted$(NC)"; \
	fi

##@ Quick Start

quickstart: ## Quick setup for CPU-only development
	@echo "$(GREEN)═══════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  Quick Start - CPU Setup$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════════$(NC)"
	$(MAKE) check-python
	$(MAKE) install-cpu
	$(MAKE) test
	@echo ""
	@echo "$(GREEN)✓ Setup complete! You can now run:$(NC)"
	@echo "  $(CYAN)python -m retrieval_augmented_generation.rag_and_qa$(NC)"

quickstart-gpu: ## Quick setup for GPU development
	@echo "$(GREEN)═══════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  Quick Start - GPU Setup$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════════$(NC)"
	$(MAKE) check-python
	$(MAKE) check-cuda
	$(MAKE) install
	$(MAKE) test-gpu
	@echo ""
	@echo "$(GREEN)✓ Setup complete! GPU acceleration enabled$(NC)"

##@ Documentation

info: ## Display project information
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  LLMs for Compliance Analysis$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(YELLOW)Project Structure:$(NC)"
	@echo "  • finetuning/                 - Model fine-tuning"
	@echo "  • synthetic_dataset_creation/ - Dataset generation"
	@echo "  • retrieval_augmented_generation/ - RAG systems"
	@echo "  • data/                       - GDPR & AI Act data"
	@echo "  • notebooks/                  - Jupyter notebooks"
	@echo ""
	@echo "$(YELLOW)Main Scripts:$(NC)"
	@echo "  • python -m finetuning.finetuning_with_unsloth"
	@echo "  • python -m synthetic_dataset_creation.synthetic_data_generation"
	@echo "  • python -m retrieval_augmented_generation.rag_and_qa"
	@echo ""
	@echo "$(YELLOW)Documentation:$(NC)"
	@echo "  • INIT_LLM_DOCS.md           - LLM initialization guide"
	@echo "  • DATAPIZZA_RAG_GUIDE.md     - DatapizzaRAG usage"
	@echo "  • README.md                  - Project overview"
	@echo ""
