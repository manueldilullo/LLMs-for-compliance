"""
Generation evaluation pipeline.
"""

import os
import gc
import asyncio
import logging
import pandas as pd
from typing import List, Dict, Any, Callable, Optional
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None

from .metrics import compute_all_metrics
from ..utils import compute_config_hash, load_checkpoint, save_checkpoint, get_timestamp

logger = logging.getLogger(__name__)


async def evaluate_model_pattern(
    model_name: str,
    pattern_name: str,
    pattern_func: Callable,
    llm_func: Callable,
    rags_dict: Optional[Any],
    domain: str,
    qa_pairs: List[Dict],
    topk: int,
    topk_max: int,
    limit: Optional[int] = None
) -> tuple:
    """Evaluate a single model×pattern×topk combination."""
    
    logger.info(f"START evaluate_model_pattern: {model_name} | {pattern_name} | topk={topk}")
    
    predictions = []
    references = []
    
    rag = rags_dict[domain]  # domain specific rag
    
    qa_subset = qa_pairs[:limit] if limit else qa_pairs
    logger.info(f"Processing {len(qa_subset)} samples...")
    
    for i, qa in enumerate(tqdm(qa_subset, desc=f"{model_name}|{pattern_name}|k={topk}")):
        query = qa["question"]
        
        reference = str(qa.get("answer", "")).strip()
        if not reference:
            reference = "No reference available"
        
        logger.debug(f"Sample {i}: calling pattern...")
        
        try:
            # Call pattern-specific function with topk
            if pattern_name == "baseline":
                pred = await asyncio.wait_for(
                    pattern_func(query, llm_func),
                    timeout=60.0
                )
            elif pattern_name == "rag_routing":
                pred = await asyncio.wait_for(
                    pattern_func(model_name, query, rags_dict, llm_func, topk=topk, topk_max=topk_max),
                    timeout=60.0
                )
            else:
                pred = await asyncio.wait_for(
                    pattern_func(model_name, query, rag, llm_func, topk=topk, topk_max=topk_max),
                    timeout=60.0
                )
            predictions.append(pred)
            references.append(reference)
            
            logger.debug(f"Sample {i}: got prediction type={type(pred)}, value={pred[:50] if isinstance(pred, str) else pred}")
            
        except asyncio.TimeoutError:
            logger.warning(f"Timeout on sample {i}")
            continue
        except Exception as e:
            logger.error(f"Error on sample {i}: {e}")
            continue
    
    # Creating predictions dataframe
    df_predictions = pd.DataFrame({
        "question": [qa["question"] for qa in qa_subset],
        "prediction": predictions,
        "reference": references,
        "model": [model_name] * len(predictions),
        "pattern": [pattern_name] * len(predictions),
        "topk": [topk] * len(predictions),
    })
    
    # Compute metrics
    logger.info(f"Computing metrics on {len(predictions)} predictions...")
    for i in range(len(predictions)):
        if isinstance(predictions[i], dict):
            predictions[i] = predictions[i].get("answer", "")
        if not isinstance(predictions[i], str):
            predictions[i] = str(predictions[i])
    
    metrics = compute_all_metrics(predictions, references)
    
    result = {
        "model": model_name,
        "pattern": pattern_name,
        "topk": topk,
        "n_samples": len(predictions),
        **metrics
    }
    
    logger.info(f"✅ {model_name} | {pattern_name} | topk={topk} | F1={metrics['f1']:.3f} | ROUGE-L={metrics['rougeL']:.3f}")
    
    return result, df_predictions


async def run_full_evaluation(
    models: List[str],
    rags_dict: Any,
    domain: str,
    qa_pairs: List[Dict],
    model_dic: Dict,
    patterns: Dict[str, Callable],
    topk_values: List[int] = [1, 5, 10],
    output_dir: str = "./evaluation_results",
    limit_per_model: Optional[int] = 50,
    resume: bool = False
) -> pd.DataFrame:
    """
    Run complete evaluation grid: models × patterns × topk
    
    Args:
        models: List of model names to test
        rags_dict: Dictionary with rags per domain
        domain: String representing the domain that is being tested
        qa_pairs: Q&A dataset
        model_dic: Dictionary with model configurations
        patterns: Dictionary of pattern functions
        topk_values: List of topk values to test (default: [1, 5, 10])
        output_dir: Where to save results
        limit_per_model: Max samples per model (for quick testing)
        resume: Whether to resume from checkpoint
    """
    from ..utils import init_llm
    
    timestamp = get_timestamp()
    
    checkpoint_dir = output_dir
    checkpoint = load_checkpoint(checkpoint_dir) if resume else {}
    executions = checkpoint.get("executions", [])
    if not executions:
        checkpoint["executions"] = [timestamp]
    else:
        checkpoint["executions"].append(timestamp)
    logger.info(f"Checkpoint: {checkpoint}")
    
    output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate models exist in model_dic
    invalid_models = [m for m in models if m not in model_dic]
    if invalid_models:
        raise ValueError(f"Unknown models: {invalid_models}. Available: {list(model_dic.keys())}")
    
    # Validate topk values
    if not all(k > 0 for k in topk_values):
        raise ValueError(f"All topk values must be > 0, got: {topk_values}")
    
    all_results = []
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"🚀 Testing Model: {model_name}")
        print(f"{'='*60}\n")
        
        # Initialize model
        try:
            llm_func, llm = init_llm(model_name, model_dic)
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            continue
        
        # Test all patterns × topk combinations
        for pattern_name, pattern_func in patterns.items():
            # Baseline doesn't use RAG, so test only once
            if pattern_name == "baseline":
                topk_list = [topk_values[0]]  # Use first topk value as placeholder
            else:
                topk_list = topk_values
            
            topk_max = max(topk_list)
            
            for topk in topk_list:
                config_id = compute_config_hash(model_name, pattern_name, topk, limit_per_model)
                
                if config_id in checkpoint:
                    logger.info(f"⏭️ Skipping {model_name} | {pattern_name} | topk={topk}")
                    continue
                
                try:
                    result, df_predictions = await evaluate_model_pattern(
                        model_name=model_name,
                        pattern_name=pattern_name,
                        pattern_func=pattern_func,
                        llm_func=llm_func,
                        rags_dict=rags_dict,
                        domain=domain,
                        qa_pairs=qa_pairs,
                        topk=topk,
                        topk_max=topk_max,
                        limit=limit_per_model
                    )
                    all_results.append(result)
                    
                    # Save incremental results
                    df_results = pd.DataFrame(all_results)
                    df_results.to_csv(f"{output_dir}/results_incremental.csv", index=False)
                    
                    df_predictions.to_csv(f"{output_dir}/predictions_incremental.csv", index=False)
                    
                    checkpoint[config_id] = True
                    save_checkpoint(checkpoint_dir, checkpoint)
                except Exception as e:
                    logger.error(f"Failed {model_name} | {pattern_name} | topk={topk}: {e}")
                    continue
        
        # Free GPU memory
        del llm
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final results
    df_final = pd.DataFrame(all_results)
    if not df_final.empty:
        # Summary by pattern (average across models)
        print(f"\n{'='*60}")
        print(f"📊 Best Configurations")
        print(f"{'='*60}\n")
        best_by_pattern = df_final.loc[df_final.groupby('pattern')['f1'].idxmax()]
        print(best_by_pattern[['pattern', 'model', 'topk', 'f1', 'rougeL']].to_string(index=False))
        
        # Summary by topk (for RAG patterns only)
        print(f"\n{'='*60}")
        print(f"📊 TopK Impact (RAG patterns only)")
        print(f"{'='*60}\n")
        rag_only = df_final[df_final['pattern'].str.contains('rag')]
        if not rag_only.empty:
            topk_impact = rag_only.groupby('topk')[['f1', 'rougeL', 'bertscore_f1']].mean()
            print(topk_impact.round(3))
        
        # Summary by topk
        print(f"\n{'='*60}")
        print(f"📊 Summary by TopK")
        print(f"{'='*60}\n")
        summary = df_final.groupby(['pattern', 'topk'])[['f1', 'rougeL', 'bertscore_f1']].mean()
        print(summary.round(3))
        
        print(f"\n{'='*60}")
        print(f"✅ Evaluation Complete! Results saved to {output_dir}")
        print(f"{'='*60}\n")
    
    return df_final


def load_and_explore_results(checkpoint_dir: str = "path/to/checkpoint_file") -> pd.DataFrame:
    """
    Load and explore evaluation results from multiple executions.
    
    Args:
        checkpoint_dir: Directory containing checkpoint.json and result CSVs
        
    Returns:
        Combined DataFrame from all executions
    """
    import os
    import json
    
    def load_final_csv(checkpoint_dir: str):
        checkpoint_json = load_checkpoint(checkpoint_dir)
        executions_paths = [
            os.path.join(checkpoint_dir, execution, "results_incremental.csv")
            for execution in checkpoint_json.get("executions", [])
        ]
        
        df_final = pd.DataFrame()
        for exec_path in executions_paths:
            if os.path.exists(exec_path):
                df_exec = pd.read_csv(exec_path)
                df_final = pd.concat([df_final, df_exec])
        
        return df_final, executions_paths
    
    # Load CSV
    df_final, csv_paths = load_final_csv(checkpoint_dir)
    
    # Basic inspection
    print("\n" + "=" * 60)
    print("📂 Loaded DataFrame")
    print("=" * 60 + "\n")
    print(f"Paths: {csv_paths}")
    print(f"Shape: {df_final.shape}")
    print("\nColumns:", list(df_final.columns))
    print("\nHead:\n", df_final.head())
    
    # Value counts for key categorical columns
    if "pattern" in df_final.columns:
        print("\n" + "=" * 60)
        print("📊 Pattern distribution")
        print("=" * 60 + "\n")
        print(df_final["pattern"].value_counts())
    
    if "model" in df_final.columns:
        print("\n" + "=" * 60)
        print("📊 Model distribution")
        print("=" * 60 + "\n")
        print(df_final["model"].value_counts())
    
    if "topk" in df_final.columns:
        print("\n" + "=" * 60)
        print("📊 TopK distribution")
        print("=" * 60 + "\n")
        print(df_final["topk"].value_counts().sort_index())
    
    # PIVOT TABLE
    print("\n" + "=" * 120)
    print("📊 PIVOT TABLE: Model x Method (All Metrics)")
    print("=" * 120 + "\n")
    
    method_col = 'pattern' if 'pattern' in df_final.columns else None
    
    if method_col and 'model' in df_final.columns:
        pivot_data = df_final.pivot_table(
            index='model',
            columns=method_col,
            values=['em', 'f1', 'bleu', 'rouge1', 'rouge2', 'rougeL', 'meteor', 'bertscore_f1'],
            aggfunc='mean',
            fill_value=0
        ).round(3)
        
        print(pivot_data.to_string())
    else:
        print("Cannot create pivot: missing 'model' or 'pattern' columns.")
    
    # Best configuration per pattern
    print("\n" + "=" * 60)
    print("🏆 Best Configurations per Pattern (by F1)")
    print("=" * 60 + "\n")
    
    if 'pattern' in df_final.columns and 'f1' in df_final.columns:
        best_by_pattern = df_final.loc[df_final.groupby("pattern")["f1"].idxmax()]
        cols_best = [
            c for c in [
                "pattern", "model", "topk",
                "em", "f1", "bleu", "rouge1", "rouge2", "rougeL",
                "meteor", "bertscore_f1"
            ] if c in best_by_pattern.columns
        ]
        print(best_by_pattern[cols_best].to_string(index=False))
    
    # Summary by (pattern, topk)
    print("\n" + "=" * 60)
    print("📊 Summary by Pattern & TopK (All Metrics)")
    print("=" * 60 + "\n")
    
    group_keys = []
    if "pattern" in df_final.columns:
        group_keys.append("pattern")
    if "topk" in df_final.columns:
        group_keys.append("topk")
    
    all_metric_cols = [
        c for c in [
            "em", "f1", "bleu", "rouge1", "rouge2", "rougeL",
            "meteor", "bertscore_f1"
        ] if c in df_final.columns
    ]
    
    if group_keys and all_metric_cols:
        summary_all = df_final.groupby(group_keys)[all_metric_cols].mean()
        print(summary_all.round(3))
    
    # Correlations between metrics
    if all_metric_cols:
        print("\n" + "=" * 60)
        print("📈 Metric Correlations")
        print("=" * 60 + "\n")
        corr = df_final[all_metric_cols].corr()
        print(corr.round(3))
    
    # Save final CSV
    print("\n" + "=" * 60)
    print(f"✅ Exploration Complete! Loaded from {csv_paths}")
    
    final_csv_path = os.path.join(checkpoint_dir, "results_final.csv")
    df_final.to_csv(final_csv_path, index=False)
    print(f"✅ Final dataframe saved in: {final_csv_path}")
    print("=" * 60 + "\n")
    
    return df_final
