# LLMs for Compliance

This repository contains a production-ready, modular Python package for LLM-based compliance analysis on legal regulations (GDPR, AI Act).

## 📄 Research Summary

This work addresses the compliance challenges faced by SMEs under GDPR and the AI Act. Building on prior research on AI Act compliance, we expand to GDPR by constructing a deterministic Knowledge Graph of 99 Articles and 173 Recitals, generating over 5,000 synthetic question-answer pairs. We developed a domain-agnostic retrieval system and evaluated Agentic AI architectures (Routing, Collaborative Debate, Self-Refinement) for legal reasoning using lightweight LLMs on consumer hardware.

**📥 Download thesis:** [Link to be added]

## ⚡ Quick Commands

```bash
# View all setup options
python quick_reference.py

# Quick setup (CPU)
make install-cpu                    # or: python setup.py install-cpu

# Quick setup (GPU)
make install-gpu                    # or: python setup.py install-gpu

# Check system
make check-gpu                      # or: python setup.py check

# Run modules
python -m retrieval_augmented_generation.rag_and_qa
python -m finetuning.finetuning_with_unsloth
python -m synthetic_dataset_creation.synthetic_data_generation
```

## 📦 Package Structure

```
LLMs-for-compliance/
├── finetuning/                    # LLM fine-tuning with Unsloth
│   ├── config/                   # Configuration settings
│   ├── models/                   # Model loading and LoRA setup
│   ├── datasets/                 # Dataset loading and preparation
│   ├── training/                 # Training utilities
│   ├── evaluation/               # Evaluation metrics
│   └── utils/                    # Utility functions
│
├── synthetic_dataset_creation/    # Synthetic Q&A dataset generation
│   ├── prompts/                  # Prompt templates
│   ├── schemas/                  # Pydantic validation schemas
│   ├── config/                   # Configuration dataclasses
│   ├── pipeline/                 # Async generation pipeline
│   ├── utils/                    # Utility functions
│   └── gdpr_to_markdown/         # GDPR markdown parsing
│
├── retrieval_augmented_generation/  # RAG system for Q&A
│   ├── config/                   # Configuration settings
│   ├── prompts/                  # RAG prompt templates
│   ├── schemas/                  # Pydantic schemas
│   ├── rag/                      # RAG implementations
│   ├── patterns/                 # Agentic RAG patterns
│   ├── evaluation/               # Evaluation metrics
│   └── utils/                    # Utility functions
│
├── data/                         # Data directory
│   ├── GDPR/
│   └── AI ACT/
│
└── requirements.txt              # Package dependencies
```

## 🚀 Quick Start

### Installation

#### Option 1: Using Makefile (Linux/Mac/WSL)

```bash
# Auto-detect and install (recommended)
make install

# CPU-only installation
make install-cpu

# GPU installation (requires CUDA)
make install-gpu

# Install optional packages (Datapizza, Unsloth)
make install-optional

# Check system and GPU
make check-gpu

# Run tests
make test
```

#### Option 2: Using Python Script (Cross-platform)

```bash
# Auto-detect and install
python setup.py install

# CPU-only installation
python setup.py install-cpu

# GPU installation
python setup.py install-gpu

# Check system
python setup.py check

# Test installation
python setup.py test
```

#### Option 3: Interactive Setup (Windows)

```bash
# Windows batch script
setup.bat

# Or Linux/Mac shell script
./setup.sh
```

#### Option 4: Manual Installation

```bash
pip install -r requirements.txt

# For GPU support (CUDA 12.x)
pip install faiss-gpu-cu12
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native" \
  FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --no-cache-dir

# For CPU only
pip install faiss-cpu llama-cpp-python
```

### GPU Requirements

For GPU-accelerated inference and training:

**Prerequisites:**
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.8 or 12.x
- cuDNN (installed with CUDA)

**Check your system:**
```bash
# Using Makefile
make check-cuda
make check-gpu

# Using Python script
python setup.py check
```

**Install CUDA:**
- [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)
- For Windows: Follow installer
- For Linux: `sudo apt install nvidia-cuda-toolkit`

**Verify CUDA installation:**
```bash
nvcc --version
nvidia-smi
```

## 📚 Module Details

### 1. Finetuning Module

Fine-tune LLMs on legal Q&A datasets using Unsloth with LoRA adapters.

**Key Components**:
- `config`: Dataset paths, model constants, LoRA target modules
- `models`: Model loading, LoRA setup, inference loading
- `datasets`: Q&A dataset loading, ChatML formatting
- `training`: SFTTrainer creation, model saving
- `evaluation`: Prediction extraction, BLEU/ROUGE/METEOR/BERTScore metrics

**Running the Pipeline**:

The complete fine-tuning pipeline (`finetuning_with_unsloth.py`) performs:
1. Model and tokenizer loading with 4-bit quantization
2. LoRA adapter configuration and setup
3. Dataset loading and preprocessing (train/val/test splits)
4. Training with SFTTrainer
5. Model saving (LoRA adapters + merged model)
6. Evaluation with comprehensive metrics

```bash
# Run complete fine-tuning pipeline
python -m finetuning.finetuning_with_unsloth

# The script will:
# - Load model: unsloth/Llama-3.2-3B-Instruct-bnb-4bit
# - Setup LoRA with r=16, alpha=16
# - Train on GDPR Q&A dataset
# - Save to: ./models/finetuned_model/
# - Output: BLEU, ROUGE, METEOR, BERTScore metrics
```

**Configuration** (edit in `finetuning/config/settings.py`):
- `SELECTED_MODEL`: Choose base model
- `GDPR_QA_DATASET` / `AI_ACT_QA_DATASET`: Dataset paths
- `LORA_R`, `LORA_ALPHA`: LoRA hyperparameters
- `MAX_SEQ_LENGTH`: Sequence length (default: 2048)

### 2. Synthetic Dataset Creation Module

Generate synthetic Q&A datasets for GDPR and AI Act using LLMs.

**Key Components**:
- `prompts`: Question and answer generation templates (unity and binding types)
- `schemas`: Pydantic validation schemas for structured outputs
- `pipeline`: AsyncQAGDPRPipeline for concurrent generation with resume capability
- `utils`: Data loading, LLM wrapper creation, dataset truncation
- `gdpr_to_markdown`: GDPR markdown file parsing utilities

**Running the Pipeline**:

The complete generation pipeline (`synthetic_data_generation.py`) performs:
1. LLM initialization (llama.cpp with GGUF models)
2. GDPR knowledge graph loading
3. Async question generation across multiple types:
   - Article unity questions (single article focus)
   - Recital unity questions (single recital focus)
   - Annex unity questions (single annex focus)
   - Binding questions (article-recital relationships)
   - Binding questions (annex-article relationships)
   - Binding questions (annex-recital relationships)
4. Dataset augmentation and deduplication
5. Automatic resume from checkpoints

```bash
# Run complete synthetic data generation pipeline
python -m synthetic_dataset_creation.synthetic_data_generation

# The script will:
# - Initialize llama.cpp LLM (e.g., Llama-3.2-3B-Instruct)
# - Load GDPR graph from: data/GDPR/datasets/gdpr_w_annexes.json
# - Generate 5000+ Q&A pairs asynchronously
# - Save to: data/GDPR/datasets/synthetic_qa_dataset.jsonl
# - Support resume: Ctrl+C safe, auto-checkpoint
```

**Configuration** (edit in `synthetic_dataset_creation/config/settings.py`):
- `MODEL_NAME`: LLM to use for generation
- `INPUT_JSON`: GDPR/AI Act source file
- `OUTPUT_PATH`: Where to save generated dataset
- `LIMIT_PER_TYPE`: Max questions per category
- `TEMPERATURE`: LLM generation temperature

### 3. Retrieval Augmented Generation Module

Complete RAG system with graph enhancement and agentic patterns.

**Key Components**:
- `rag`: AbstractRAG base class, BasicRAG (FAISS), DatapizzaRAG (Qdrant)
- `patterns`: 6 agentic patterns for legal reasoning
- `evaluation`: Comprehensive metrics (EM, F1, BLEU, ROUGE, METEOR, BERTScore)
- `utils`: Caching, JSON parsing, LLM calls, document loading

**Running the Pipeline**:

The complete RAG evaluation pipeline (`rag_and_qa.py`) performs:
1. RAG system initialization (GDPR and AI Act)
2. Q&A dataset loading
3. Retrieval evaluation (with/without graph enhancement)
4. Retrieval metrics calculation (precision, recall, F1)
5. LLM initialization for generation
6. Multi-pattern generation evaluation:
   - Baseline (no RAG)
   - RAG (vector retrieval)
   - RAG+Graph (vector + knowledge graph)
   - Routing (multi-domain classification)
   - Collaboration (generator-critic debate)
   - Self-Refinement (iterative improvement)
7. Generation metrics computation (BLEU, ROUGE, METEOR, BERTScore)

```bash
# Run complete RAG pipeline
python -m retrieval_augmented_generation.rag_and_qa

# The script will:
# Step 1: Initialize BasicRAG for GDPR and AI Act
# Step 2: Load Q&A evaluation datasets
# Step 3: Run retrieval evaluation (topk=20)
#   - Basic RAG retrieval
#   - RAG with graph enhancement
#   - Cache results to: data/*/retrieval_results/*.jsonl
# Step 4: Initialize LLM (e.g., Llama-3.2-3B-Instruct)
# Step 5: Evaluate all 6 agentic patterns
#   - Test each pattern with different topk values (1, 5, 10)
#   - Cache generation results for each configuration
# Step 6: Compute and display metrics
#   - Retrieval: P@K, R@K, F1@K
#   - Generation: BLEU, ROUGE-L, METEOR, BERTScore
```

**Configuration** (edit in `retrieval_augmented_generation/rag_and_qa.py` or `config/settings.py`):
- `USE_CACHE`: Load existing shuffled datasets
- `TOPK`: Number of documents to retrieve (default: 20)
- `selected_emb`: Embedding model ("bge", "bge-large", etc.)
- `TOPK_GEN`: Top-k for generation evaluation
- `LIMIT_GEN`: Limit samples for quick testing

## 🔍 RAG Implementations

The framework provides multiple RAG implementations optimized for legal documents:

### BasicRAG (FAISS)
- **Vector Store**: FAISS (in-memory, CPU/GPU)
- **Embeddings**: SentenceTransformers
- **Features**: Fast retrieval, graph-based enhancement
- **Best For**: Development, small to medium datasets

### DatapizzaRAG (Qdrant)
- **Vector Store**: Qdrant (persistent, scalable)
- **Embeddings**: FastEmbed
- **Features**: Production-ready, metadata filtering, hybrid search
- **Best For**: Production, large datasets, distributed systems

Both implementations support:
- ✅ Knowledge graph integration (Articles ↔ Recitals ↔ Annexes)
- ✅ Retrieval caching and optimization
- ✅ Multiple embedding models (BGE, E5, multilingual)
- ✅ Configurable top-k and similarity thresholds

**Example**:
```python
from retrieval_augmented_generation import RAGFactory

# FAISS-based RAG
basic_rag = RAGFactory.create(rag_type="basic", embedding_model="BAAI/bge-base-en-v1.5")

# Qdrant-based RAG
datapizza_rag = RAGFactory.create(
    rag_type="datapizza",
    embedding_model="BAAI/bge-base-en-v1.5",
    vectorstore_path="./qdrant_storage"
)
```

## 🤖 Agentic Patterns

Six agentic AI patterns for enhanced legal reasoning:

### 1. Baseline Pattern
Simple LLM inference without retrieval. Useful for comparison and zero-shot evaluation.

### 2. RAG Pattern
**LLM + Vector Retrieval**: Retrieves relevant legal documents and generates answers based on context.

### 3. RAG + Graph Pattern
**LLM + Vector + Knowledge Graph**: Expands retrieval using document relationships (Articles cite Recitals, Annexes reference Articles).

### 4. Routing Pattern
**Domain Classification → Specialized RAG**: Routes queries to domain-specific RAGs (GDPR vs AI Act) based on content.

```python
# Automatically routes to GDPR or AI Act RAG
answer = await routing_rag_pattern(
    query="What is personal data processing?",
    rags_dict={"GDPR": gdpr_rag, "AIACT": aiact_rag},
    llm_func=your_llm
)
```

### 5. Collaboration Pattern (Debate)
**Generator ↔ Critic Loop**: A generator produces an answer, a critic identifies issues, and the process iterates until consensus or max rounds.

```python
# Multi-round refinement
answer = await collaboration_rag_pattern(
    query="Who is the data controller?",
    rag=rag,
    llm_func=your_llm,
    max_rounds=3
)
```

### 6. Self-Refinement Pattern
**Iterative Self-Improvement**: The LLM critiques its own answers and refines them over multiple iterations.

```python
# Self-critique and improvement
answer = await self_refinement_rag_pattern(
    query="What are the data subject rights?",
    rag=rag,
    llm_func=your_llm,
    max_iterations=2
)
```

**Pattern Comparison**:
| Pattern | Retrieval | Graph | Multi-Agent | Iterations | Use Case |
|---------|-----------|-------|-------------|------------|----------|
| Baseline | ❌ | ❌ | ❌ | 1 | Zero-shot baseline |
| RAG | ✅ | ❌ | ❌ | 1 | Standard retrieval |
| RAG+Graph | ✅ | ✅ | ❌ | 1 | Connected legal concepts |
| Routing | ✅ | ❌ | ✅ | 1 | Multi-domain queries |
| Collaboration | ✅ | ❌ | ✅ | N | Complex reasoning |
| Self-Refinement | ✅ | ❌ | ❌ | N | Answer quality improvement |

## 🔬 Key Features

### Modular Design
- **Independent Modules**: Each component (finetuning, synthetic data, RAG) can be used standalone
- **Clear Separation**: Config, models, data, training, and evaluation are cleanly separated
- **Extensible**: Easy to add new patterns, RAG implementations, or evaluation metrics

### Production-Ready
- **Type Safety**: Comprehensive type hints throughout the codebase
- **Error Handling**: Robust error handling and validation using Pydantic
- **Logging**: Structured logging for debugging and monitoring
- **Resume Capability**: Long-running processes (synthetic data generation) support checkpointing

### Legal Domain Optimized
- **Knowledge Graphs**: Deterministic graphs of Articles ↔ Recitals ↔ Annexes relationships
- **Multi-Regulation**: Handles GDPR (99 Articles, 173 Recitals) and AI Act seamlessly
- **Domain Routing**: Automatic classification and routing to appropriate regulation
- **Legal Metrics**: Specialized evaluation for legal Q&A (exact match, F1, semantic similarity)

### Scalable & Efficient
- **Async Processing**: Concurrent generation for synthetic datasets
- **Smart Caching**: Retrieval and generation results cached for efficiency
- **GPU Acceleration**: Full CUDA support for FAISS, llama-cpp-python, and PyTorch
- **Consumer Hardware**: Optimized for lightweight LLMs (3B-8B parameters) on consumer GPUs

### Flexible Configuration
- **Multiple Embeddings**: BGE, E5, multilingual models supported
- **LoRA Adapters**: Efficient fine-tuning with configurable rank and alpha
- **Customizable Patterns**: Easy to modify or add new agentic patterns
- **Prompt Engineering**: All prompts externalized and configurable

## 📝 Requirements

**Python**: 3.8+

**Core Dependencies**:
- `torch >= 2.0.0` - Deep learning framework
- `transformers >= 4.36.0` - Hugging Face models
- `sentence-transformers >= 2.2.0` - Embeddings
- `pydantic >= 2.0.0` - Data validation

**Fine-tuning**:
- `unsloth` - Efficient LoRA training
- `peft >= 0.7.0` - Parameter-efficient fine-tuning
- `trl >= 0.7.0` - Transformer Reinforcement Learning

**RAG**:
- `faiss-cpu` or `faiss-gpu-cu12 >= 1.7.4` - Vector search
- `llama-cpp-python >= 0.2.0` - Local LLM inference
- `networkx >= 3.0` - Knowledge graph

**Evaluation**:
- `evaluate >= 0.4.0` - Hugging Face metrics
- `bert-score >= 0.3.13` - Semantic similarity
- `rouge-score >= 0.1.2` - Summarization metrics

**Optional**:
- `datapizza-ai` - Qdrant-based RAG (production)
- `qdrant-client` - Vector database client

See [`requirements.txt`](requirements.txt) for complete list.

**Installation**:
```bash
# Quick setup
make install              # Auto-detect GPU/CPU
python setup.py install   # Cross-platform

# See "Quick Start" section for detailed instructions
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🎓 Citation

If you use this framework in your research, please cite:

```bibtex
@mastersthesis{llms_compliance_2026,
  title={Agentic AI for AI Act and GDPR compliance},
  author={Manuel Di Lullo},
  year={2025},
  school={Sapienza University of Rome},
  type={Master's Thesis}
}
```

## 📧 Contact

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Institution**: Sapienza University of Rome
- **Research Line**: Legal AI, Compliance Automation, RAG Systems

---

**Note**: This research builds upon [Tommaso Sgroi's thesis](https://example.com/sgroi-thesis) on AI Act compliance using graph-based RAG.
