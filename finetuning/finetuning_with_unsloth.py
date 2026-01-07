"""
Fine-tuning LLM with Unsloth

This script demonstrates fine-tuning language models using the Unsloth library
with LoRA adapters for efficient training on legal Q&A datasets (GDPR/AI Act).

Usage:
    python -m finetuning.finetuning_with_unsloth

The script:
1. Loads a pre-trained model with 4-bit quantization
2. Sets up LoRA adapters for efficient fine-tuning
3. Prepares the Q&A dataset
4. Trains with early stopping and evaluation
5. Saves both LoRA adapters and merged model
6. Evaluates on test set with multiple metrics
"""

# Import modularized components
from finetuning import (
    # Configuration
    AI_ACT_QA_DATASET,
    GDPR_QA_DATASET,
    
    # Model functions
    load_model_and_tokenizer,
    setup_lora_adapters,
    load_model_for_inference,
    
    # Dataset functions
    load_qa_dataset,
    prepare_datasets,
    
    # Training functions
    create_trainer,
    save_model,
    
    # Evaluation functions
    safe_extract_predictions,
    compute_metrics,
    
    # Utilities
    clear_gpu_memory,
    print_gpu_info,
)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main entry point for fine-tuning pipeline."""
    # Configuration
    MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    DATASET_PATH = AI_ACT_QA_DATASET  # or GDPR_QA_DATASET
    OUTPUT_DIR = ".results/aiact/llama_unsloth_qa"
    LORA_OUTPUT = "llama-3.2-unsloth-qa-lora-ai-act"
    MERGED_OUTPUT = "llama-3.2-unsloth-qa-ai-act-merged"

    # Print GPU info
    print("=" * 60)
    print("GPU Information")
    print("=" * 60)
    print_gpu_info()
    clear_gpu_memory()

    # Load model and tokenizer
    print("\n" + "=" * 60)
    print("Step 1: Loading Model and Tokenizer")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # Setup LoRA adapters
    print("\n" + "=" * 60)
    print("Step 2: Setting up LoRA Adapters")
    print("=" * 60)
    model = setup_lora_adapters(model)

    # Load and prepare dataset
    print("\n" + "=" * 60)
    print("Step 3: Loading and Preparing Dataset")
    print("=" * 60)
    print(f"Dataset: {DATASET_PATH}")
    train, val, test, test_raw = load_qa_dataset(DATASET_PATH)
    train, val, test = prepare_datasets(train, val, test, tokenizer)

    print(f"\nSample training text:\n{train[0]['text'][:400]}...\n")

    # Create trainer
    print("\n" + "=" * 60)
    print("Step 4: Creating Trainer")
    print("=" * 60)
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train,
        val_dataset=val,
        output_dir=OUTPUT_DIR,
    )

    # Train
    print("\n" + "=" * 60)
    print("Step 5: Training")
    print("=" * 60)
    trainer.train(resume_from_checkpoint=True)

    # Save model
    print("\n" + "=" * 60)
    print("Step 6: Saving Model")
    print("=" * 60)
    save_model(trainer, tokenizer, LORA_OUTPUT, MERGED_OUTPUT)

    # Evaluation
    print("\n" + "=" * 60)
    print("Step 7: Evaluation")
    print("=" * 60)
    model, tokenizer = load_model_for_inference(MODEL_NAME, LORA_OUTPUT)
    predictions, references = safe_extract_predictions(test_raw, model, tokenizer)
    results = compute_metrics(predictions, references)

    print("\nTest Set Results:")
    print("=" * 60)
    for metric, value in results.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, (int, float)):
                    print(f"{metric}.{k}: {v:.4f}")
        elif isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")

    # Show samples
    print("\n" + "=" * 60)
    print("Sample Predictions")
    print("=" * 60)
    for i in range(min(5, len(predictions))):
        print(f"\n[Sample {i+1}]")
        print(f"Question: {test_raw[i]['messages'][0]['content'][:80]}...")
        print(f"Predicted: {predictions[i][:100]}...")
        print(f"Reference: {references[i][:100]}...")

    print("\n" + "=" * 60)
    print("Fine-tuning Complete!")
    print("=" * 60)
    print(f"LoRA adapters saved to: {LORA_OUTPUT}")
    print(f"Merged model saved to: {MERGED_OUTPUT}")


if __name__ == "__main__":
    main()
