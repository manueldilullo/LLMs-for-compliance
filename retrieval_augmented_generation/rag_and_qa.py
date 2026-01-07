"""
Retrieval Augmented Generation (RAG) and Q&A System

This script demonstrates the complete RAG pipeline for legal Q&A including
vector retrieval, graph enhancement, and multiple agentic patterns.

Usage:
    python -m retrieval_augmented_generation.rag_and_qa

The script:
1. Initializes RAG systems for GDPR and AI Act
2. Loads Q&A evaluation datasets
3. Runs retrieval evaluation with and without graph enhancement
4. Calculates retrieval metrics (precision, recall, F1)
5. Caches results for later generation evaluation
"""

import os
import json
import numpy as np

# Import modularized components
from retrieval_augmented_generation import (
    # Configuration
    Config,
    GRAPH_FILENAMES,
    EMBEDDING_MODELS,
    
    # RAG components
    init_rag,
    RAGFactory,
    
    # Evaluation
    calculate_retrieval,
    
    # Utilities
    load_dataset,
)


# =============================================================================
# Configuration
# =============================================================================

# Dataset configuration
USE_CACHE = True  # Load existing shuffled datasets
TOPK = 20  # Number of documents to retrieve

# GDPR Configuration
gdpr_config = Config(
    domain="data/GDPR",
    selected_emb="bge",  # or "bge-large"
    graph_filename=GRAPH_FILENAMES["GDPR"],
)

# AI Act Configuration
aiact_config = Config(
    domain="data/AI ACT",
    selected_emb="bge",  # or "bge-large"
    graph_filename=GRAPH_FILENAMES["AI ACT"],
)


# =============================================================================
# Main Execution
# =============================================================================

import asyncio
from typing import Callable
from functools import partial
from llama_cpp import Llama
from retrieval_augmented_generation.evaluation.generation import evaluate_model_pattern
from retrieval_augmented_generation.patterns import (
    baseline_pattern,
    rag_pattern,
    rag_with_graph_pattern,
    routing_rag_pattern,
    collaboration_rag_pattern,
    self_refinement_rag_pattern
)

async def create_llama_cpp_func(llm) -> Callable:
    """
    Create an async wrapper for llama.cpp model.
    """
    async def llama_cpp_func(prompt, stop=None, **kwargs):
        loop = asyncio.get_running_loop()
        call_func = partial(
            llm,
            prompt=prompt,
            stop=stop or [],
            **kwargs
        )
        response = await loop.run_in_executor(None, call_func)
        return response['choices'][0]['text']

    return llama_cpp_func

async def main():
    """Main entry point for RAG evaluation."""
    print("=" * 60)
    print("RAG Retrieval Evaluation Pipeline")
    print("=" * 60)
    
    # Initialize RAG systems
    print("\nStep 1: Initializing RAG systems...")
    print("-" * 60)
    
    print(f"Loading GDPR RAG (embedding: {gdpr_config.SELECTED_EMB})...")
    basic_gdpr_rag = init_rag(gdpr_config, rag_type="basic")
    print(f"  - Documents: {len(basic_gdpr_rag.documents)}")
    print(f"  - Embeddings shape: {basic_gdpr_rag.embeddings.shape}")
    
    print(f"\nLoading AI Act RAG (embedding: {aiact_config.SELECTED_EMB})...")
    basic_aiact_rag = init_rag(aiact_config, rag_type="basic")
    print(f"  - Documents: {len(basic_aiact_rag.documents)}")
    print(f"  - Embeddings shape: {basic_aiact_rag.embeddings.shape}")

    # Load Q&A datasets
    print("\nStep 2: Loading Q&A datasets...")
    print("-" * 60)
    
    qa_pairs_dict = {}
    
    for config in [gdpr_config, aiact_config]:
        if USE_CACHE and os.path.exists(config.SHUFFLED_DATASET):
            print(f"Loading shuffled {config.DOMAIN} dataset from cache...")
            qa_pairs = load_dataset(config.SHUFFLED_DATASET)
        else:
            print(f"Loading and shuffling {config.DOMAIN} dataset...")
            qa_pairs = np.array(load_dataset(config.DATASET_JSONL))
            np.random.shuffle(qa_pairs)
            qa_pairs = qa_pairs.tolist()

            # Save shuffled dataset
            os.makedirs(os.path.dirname(config.SHUFFLED_DATASET), exist_ok=True)
            with open(config.SHUFFLED_DATASET, "w") as f:
                for entry in qa_pairs:
                    f.write(json.dumps(entry) + '\n')
            print(f"  Saved shuffled dataset to {config.SHUFFLED_DATASET}")

        qa_pairs_dict[config.DOMAIN] = qa_pairs
        print(f"  Loaded {len(qa_pairs)} QA pairs for {config.DOMAIN}")

    # Run retrieval evaluation
    print("\nStep 3: Running retrieval evaluation...")
    print("-" * 60)
    
    # Define RAG configurations to evaluate
    rags = {
        "gdpr_basic": {
            "rag": basic_gdpr_rag,
            "retrieval_path": os.path.join(gdpr_config.RETRIEVAL_RESULTS_DIR, "basic_rag_cache.jsonl"),
            "use_graph": False,
            "config": gdpr_config
        },
        "gdpr_basic_graph": {
            "rag": basic_gdpr_rag,
            "retrieval_path": os.path.join(gdpr_config.RETRIEVAL_RESULTS_DIR, "basic_ragg_cache.jsonl"),
            "use_graph": True,
            "config": gdpr_config
        },
        "aiact_basic": {
            "rag": basic_aiact_rag,
            "retrieval_path": os.path.join(aiact_config.RETRIEVAL_RESULTS_DIR, "basic_rag_cache.jsonl"),
            "use_graph": False,
            "config": aiact_config
        },
        "aiact_basic_graph": {
            "rag": basic_aiact_rag,
            "retrieval_path": os.path.join(aiact_config.RETRIEVAL_RESULTS_DIR, "basic_ragg_cache.jsonl"),
            "use_graph": True,
            "config": aiact_config
        },
    }

    for rag_name, rag_info in rags.items():
        rag = rag_info["rag"]
        retrieval_path = rag_info["retrieval_path"]
        retrieval_folder = os.path.dirname(retrieval_path)
        use_graph = rag_info["use_graph"]
        config = rag_info["config"]

        print(f"\n{'=' * 60}")
        print(f"RAG: {rag_name.upper()}")
        print(f"Graph enhancement: {use_graph}")
        print(f"Cache: {retrieval_path}")
        print('=' * 60)

        # Create output directory
        os.makedirs(retrieval_folder, exist_ok=True)

        # Run retrieval evaluation
        rag_results = calculate_retrieval(
            rag=rag,
            qa_pairs=qa_pairs_dict[config.DOMAIN],
            top_k=TOPK,
            use_graph=use_graph,
            threshold=float("-inf"),
            cache_file=retrieval_path,
            load_cache=USE_CACHE
        )

        print(f"\nResults for {rag_name}:")
        if isinstance(rag_results, dict):
            print(f"  Precision@{TOPK}: {rag_results.get('precision', 0):.4f}")
            print(f"  Recall@{TOPK}: {rag_results.get('recall', 0):.4f}")
            print(f"  F1@{TOPK}: {rag_results.get('f1', 0):.4f}")

    print("\n" + "=" * 60)
    print("RAG Retrieval Evaluation Complete!")
    print("=" * 60)

    # =========================================================================
    # Generation & Agentic Evaluation
    # =========================================================================
    
    print("\nStep 4: Initializing LLM for Generation...")
    print("-" * 60)
    
    # LLM Configuration
    REPO_ID = "bartowski/Llama-3.2-3B-Instruct-GGUF"
    MODEL_FILE = "Llama-3.2-3B-Instruct-f16.gguf"
    
    try:
        llm = Llama.from_pretrained(
            repo_id=REPO_ID,
            filename=MODEL_FILE,
            verbose=False,
            n_ctx=4096,
            n_gpu_layers=33,  # Adjust based on VRAM
        )
        llm_func = await create_llama_cpp_func(llm)
        print(f"LLM loaded: {REPO_ID}/{MODEL_FILE}")
    except Exception as e:
        print(f"Error loading LLM: {e}")
        return

    # Patterns to evaluate
    patterns = {
        "baseline": baseline_pattern,
        "rag": rag_pattern,
        "rag_graph": rag_with_graph_pattern,
        "rag_routing": routing_rag_pattern,
        "rag_collaboration": collaboration_rag_pattern,
        "rag_self_refinement": self_refinement_rag_pattern,
    }
    
    # RAGs dict for routing
    rags_dict = {
        "GDPR": basic_gdpr_rag,
        "AIACT": basic_aiact_rag
    }

    # Generation loop
    print("\nStep 5: Running Generation Evaluation...")
    print("-" * 60)
    
    # Evaluation config
    TOPK_GEN = 5
    LIMIT_GEN = None # Set to a small number (e.g. 5) for testing
    
    results_summary = []

    for domain_config in [gdpr_config, aiact_config]:
        domain = domain_config.DOMAIN.split("/")[-1] # "GDPR" or "AI ACT" (which matches keys in RAGs dict?)
        # Fix domain keys in rags_dict to match config domains
        # config.DOMAIN is "data/GDPR" -> "GDPR"
        domain_key = "GDPR" if "GDPR" in domain_config.DOMAIN else "AIACT"
        
        print(f"\nEvaluating Domain: {domain_key}")
        
        qa_pairs = qa_pairs_dict[domain_config.DOMAIN]
        
        for pattern_name, pattern_func in patterns.items():
            print(f"\n  Pattern: {pattern_name}")
            
            try:
                metrics, df_preds = await evaluate_model_pattern(
                    model_name="Llama-3.2-3B",
                    pattern_name=pattern_name,
                    pattern_func=pattern_func,
                    llm_func=llm_func,
                    rags_dict=rags_dict,
                    domain=domain_key,
                    qa_pairs=qa_pairs,
                    topk=TOPK_GEN,
                    topk_max=TOPK,
                    limit=LIMIT_GEN
                )
                
                # Save results
                output_file = os.path.join(domain_config.GENERATOR_RESULTS_DIR, f"results_{pattern_name}.csv")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                df_preds.to_csv(output_file, index=False)
                print(f"    Saved predictions to {output_file}")
                
                # Print metrics
                print("    Metrics:")
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                         print(f"      {k}: {v:.4f}")
                    elif isinstance(v, dict):
                         print(f"      {k}: {v}")

                results_summary.append({
                    "domain": domain_key,
                    "pattern": pattern_name,
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                })
                
            except Exception as e:
                print(f"    Error evaluating {pattern_name}: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 60)
    print("Generation Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
