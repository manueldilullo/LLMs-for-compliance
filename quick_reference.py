#!/usr/bin/env python3
"""
Quick Setup Reference - LLMs-for-compliance
Print quick reference for setup commands
"""

SETUP_REFERENCE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    LLMs-for-compliance - Setup Reference                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 QUICK START
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Linux/Mac/WSL (Makefile):
    make install                 # Auto-detect and install
    make quickstart              # Quick CPU setup
    make quickstart-gpu          # Quick GPU setup

  Cross-platform (Python):
    python setup.py install      # Auto-detect and install
    python setup.py install-cpu  # CPU-only
    python setup.py install-gpu  # GPU with CUDA

  Windows (Batch):
    setup.bat                    # Interactive menu

  Linux/Mac (Shell):
    ./setup.sh                   # Interactive menu

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔧 INSTALLATION OPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Base Installation:
    make install                 # Auto-detect GPU/CPU
    make install-cpu             # Force CPU-only
    make install-gpu             # Force GPU (requires CUDA)

  Optional Packages:
    make install-optional        # Datapizza, Unsloth
    make install-datapizza       # Datapizza AI (Qdrant RAG)
    make install-unsloth         # Unsloth (efficient fine-tuning)

  Development Tools:
    make install-dev             # Jupyter, pytest, black, etc.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 SYSTEM CHECKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Check Environment:
    make check-python            # Python version
    make check-cuda              # CUDA installation
    make check-gpu               # GPU availability

  Test Installation:
    make test                    # Core imports
    make test-gpu                # GPU functionality
    make test-modules            # Custom modules

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 MANUAL GPU SETUP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Install CUDA Toolkit:
     https://developer.nvidia.com/cuda-downloads

  2. Verify CUDA:
     nvcc --version
     nvidia-smi

  3. Install GPU packages:
     # FAISS (CUDA 12.x)
     pip install faiss-gpu-cu12

     # llama-cpp-python with CUDA
     CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native" \\
       FORCE_CMAKE=1 \\
       pip install llama-cpp-python --force-reinstall --no-cache-dir

  4. Verify GPU:
     python -c "import torch; print(torch.cuda.is_available())"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 RUNNING MODULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Fine-tuning:
    python -m finetuning.finetuning_with_unsloth

  Synthetic Dataset Creation:
    python -m synthetic_dataset_creation.synthetic_data_generation

  RAG & Q&A:
    python -m retrieval_augmented_generation.rag_and_qa

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🛠️ UTILITIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Cleanup:
    make clean                   # Remove cache files
    make clean-models            # Remove downloaded models

  Information:
    make info                    # Project info
    make help                    # Show all commands

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 PACKAGE STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  finetuning/                   - LLM fine-tuning (Unsloth)
  synthetic_dataset_creation/   - Q&A dataset generation
  retrieval_augmented_generation/ - RAG systems & patterns
  data/                         - GDPR & AI Act datasets
  notebooks/                    - Jupyter notebooks

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 TIPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • For development without GPU: make quickstart
  • For production with GPU: make quickstart-gpu
  • Test imports after install: make test
  • Check GPU before GPU install: make check-gpu
  • Install optional packages later: make install-optional

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 DOCUMENTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  README.md                     - Project overview
  INIT_LLM_DOCS.md             - LLM initialization guide
  DATAPIZZA_RAG_GUIDE.md       - DatapizzaRAG usage guide

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

if __name__ == "__main__":
    print(SETUP_REFERENCE)
