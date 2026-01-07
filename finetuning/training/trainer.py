"""
Training utilities for fine-tuning with SFTTrainer.
"""

from typing import Any

from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback


def create_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: str = ".results/llama_unsloth_qa",
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 100,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.03,
    early_stopping_patience: int = 3,
    max_seq_length: int = 2048,
    use_bf16: bool = False,
    use_fp16: bool = True,
) -> SFTTrainer:
    """
    Create an SFTTrainer for fine-tuning.

    Args:
        model: The model to train
        tokenizer: The tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_dir: Directory to save checkpoints
        per_device_train_batch_size: Training batch size per device
        per_device_eval_batch_size: Evaluation batch size per device
        gradient_accumulation_steps: Number of gradient accumulation steps
        num_train_epochs: Maximum number of training epochs
        learning_rate: Learning rate
        warmup_ratio: Warmup ratio for learning rate scheduler
        early_stopping_patience: Patience for early stopping
        max_seq_length: Maximum sequence length
        use_bf16: Whether to use bf16 precision
        use_fp16: Whether to use fp16 precision

    Returns:
        Configured SFTTrainer instance
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to="none",
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        optim="adamw_torch",
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=use_bf16,
        fp16=use_fp16,
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=True,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    return trainer


def save_model(
    trainer: SFTTrainer,
    tokenizer: Any,
    lora_output_dir: str,
    merged_output_dir: str,
) -> None:
    """
    Save LoRA adapters and merged model.

    Args:
        trainer: The trainer with the trained model
        tokenizer: The tokenizer to save
        lora_output_dir: Directory to save LoRA adapters
        merged_output_dir: Directory to save merged model
    """
    # Save LoRA adapters
    trainer.model.save_pretrained(lora_output_dir, tokenizer=tokenizer)

    # Merge and save full model
    trainer.model = trainer.model.merge_and_unload()
    trainer.model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)
