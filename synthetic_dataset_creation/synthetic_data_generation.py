"""
Synthetic Dataset Generation for GDPR and AI Act

This script generates synthetic Q&A datasets for legal regulations using LLMs
with structured prompts and Pydantic validation.

Usage:
    python -m synthetic_dataset_creation.synthetic_data_generation

The script:
1. Loads the GDPR/AI Act regulation structure
2. Creates an async LLM wrapper (llama.cpp)
3. Generates Q&A pairs for articles, recitals, and annexes
4. Generates binding questions across different sections
5. Augments the dataset with variations
6. Saves results with resume capability
"""

import asyncio
from llama_cpp import Llama

# Import modularized components
from synthetic_dataset_creation import (
    # Data loading
    load_gdpr_graph,
    
    # LLM utilities
    create_llama_cpp_func,
    truncate_dataset,
    
    # Prompts
    ALL_PROMPTS,
    
    # Pipeline
    AsyncQAGDPRPipeline,
    
    # Config
    UnityConfig,
    BindingConfig,
)


# =============================================================================
# Configuration
# =============================================================================

# Model configuration
MODEL_REPO = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF"
MODEL_FILE = "DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"

# Data paths
GDPR_DATA_PATH = "data/GDPR/datasets/gdpr_w_annexes.json"
OUTPUT_PATH = "data/GDPR/datasets/full_dataset.jsonl"

# Generation limits
LIMIT = None  # None for all, or integer to limit per section
N_REPETITIONS = 3  # Number of Q&A pairs per excerpt
N_AUGMENTATIONS = 1  # Number of augmented variations
MAX_CONCURRENCY = 10  # Async concurrency limit


# =============================================================================
# Main Execution
# =============================================================================

async def main():
    """Main entry point for synthetic dataset generation."""
    print("=" * 60)
    print("Synthetic Dataset Generation Pipeline")
    print("=" * 60)
    print(f"Model: {MODEL_REPO}/{MODEL_FILE}")
    print(f"Data: {GDPR_DATA_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Limit per section: {LIMIT or 'All'}")
    print(f"Repetitions: {N_REPETITIONS}")
    print(f"Augmentations: {N_AUGMENTATIONS}")
    print(f"Max concurrency: {MAX_CONCURRENCY}")
    print("=" * 60)

    # Initialize LLM
    print("\nStep 1: Loading LLM...")
    llm = Llama.from_pretrained(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        verbose=False,
        n_batch=512,
        n_gpu_layers=40,
        n_ctx=4096,
    )

    # Create async wrapper
    print("Step 2: Creating async LLM wrapper...")
    llama_cpp_func = await create_llama_cpp_func(llm)

    # Load data
    print(f"Step 3: Loading regulation data from {GDPR_DATA_PATH}...")
    gdpr_data = load_gdpr_graph(GDPR_DATA_PATH)
    print(f"  - Articles: {len(gdpr_data.get('articles', []))}")
    print(f"  - Recitals: {len(gdpr_data.get('recitals', []))}")
    print(f"  - Annexes: {len(gdpr_data.get('annexes', []))}")

    # Initialize pipeline
    print(f"\nStep 4: Initializing pipeline...")
    pipeline = AsyncQAGDPRPipeline(
        data=gdpr_data,
        llm_func=llama_cpp_func,
        output_path=OUTPUT_PATH,
        max_concurrency=MAX_CONCURRENCY,
        prompts=ALL_PROMPTS
    )

    # Run generation steps
    try:
        print("\n" + "=" * 60)
        print("STEP 5: Generating Article Unity Q&A")
        print("=" * 60)
        await pipeline.generate_article_unity(limit=LIMIT, n=N_REPETITIONS)

        print("\n" + "=" * 60)
        print("STEP 6: Generating Recital Unity Q&A")
        print("=" * 60)
        await pipeline.generate_recital_unity(limit=LIMIT, n=N_REPETITIONS)

        print("\n" + "=" * 60)
        print("STEP 7: Generating Annex Unity Q&A")
        print("=" * 60)
        await pipeline.generate_annex_unity(limit=LIMIT, n=N_REPETITIONS)

        print("\n" + "=" * 60)
        print("STEP 8: Generating Binding Article-Recital Q&A")
        print("=" * 60)
        await pipeline.generate_binding_article_recital_questions(limit=LIMIT, n=N_REPETITIONS)

        print("\n" + "=" * 60)
        print("STEP 9: Generating Binding Annex-Article Q&A")
        print("=" * 60)
        await pipeline.generate_binding_annex_article_questions(limit=LIMIT, n=N_REPETITIONS)

        print("\n" + "=" * 60)
        print("STEP 10: Generating Binding Annex-Recital Q&A")
        print("=" * 60)
        await pipeline.generate_binding_annex_recital_questions(limit=LIMIT, n=N_REPETITIONS)

        print("\n" + "=" * 60)
        print("STEP 11: Augmenting Dataset")
        print("=" * 60)
        await pipeline.augment_dataset(n=N_AUGMENTATIONS)

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Total records generated: {len(pipeline.dataset)}")
        print(f"Output saved to: {OUTPUT_PATH}")

        # Optional: truncate dataset
        if LIMIT is not None:
            truncated_path = OUTPUT_PATH.replace(".jsonl", "_truncated.jsonl")
            print(f"\nTruncating dataset to {LIMIT * 100} samples...")
            truncate_dataset(OUTPUT_PATH, truncated_path, limit=LIMIT * 100)
            print(f"Truncated dataset saved to: {truncated_path}")

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Pipeline stopped by user")
        print(f"Partial results saved to: {OUTPUT_PATH}")
        print(f"Records generated so far: {len(pipeline.dataset)}")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        print(f"Partial results saved to: {OUTPUT_PATH}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
