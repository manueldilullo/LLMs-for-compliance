"""
Dataset loading and formatting utilities for fine-tuning.
"""

from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset


def to_chat_example(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a Q&A item to ChatML-like message format.

    Args:
        item: Dictionary with 'question' and 'answer' keys

    Returns:
        Dictionary with 'messages' key containing user/assistant messages
    """
    q = item["question"].strip()
    a = item["answer"].strip()
    return {
        "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
    }


def load_qa_dataset(
    json_path: str,
    shuffle: bool = True,
    test_size: float = 0.2,
    val_size: float = 0.25,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    """
    Load and split a Q&A dataset from JSONL file.

    Args:
        json_path: Path to the JSONL dataset file
        shuffle: Whether to shuffle the dataset
        test_size: Fraction of data for test set
        val_size: Fraction of remaining data for validation set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, test_raw_dataset)
    """
    # Read the JSONL file
    qa_list = pd.read_json(path_or_buf=json_path, lines=True)

    # Convert to ChatML-like structure
    chat_rows = [to_chat_example(row) for _, row in qa_list.iterrows()]

    if shuffle:
        np_chat_rows = np.array(chat_rows[:])
        np.random.shuffle(np_chat_rows)
        chat_rows = np_chat_rows.tolist()

    dataset = Dataset.from_list(chat_rows)

    # Split dataset using Hugging Face datasets' native method
    train_val_test = dataset.train_test_split(test_size=test_size, seed=seed)
    train_val = train_val_test["train"]
    test = train_val_test["test"]

    train_val_split = train_val.train_test_split(test_size=val_size, seed=seed)
    train = train_val_split["train"]
    val = train_val_split["test"]

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    return train, val, test, test  # Return test twice (processed and raw)


def formatting_prompts_func(examples: Dict[str, Any], tokenizer: Any) -> Dict[str, List[str]]:
    """
    Format messages into training strings using the tokenizer's chat template.

    Args:
        examples: Dictionary with 'messages' key containing conversation data
        tokenizer: The tokenizer with chat template

    Returns:
        Dictionary with 'text' key containing formatted strings
    """
    texts = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}


def prepare_datasets(
    train: Dataset,
    val: Dataset,
    test: Dataset,
    tokenizer: Any,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Apply formatting to all dataset splits.

    Args:
        train: Training dataset
        val: Validation dataset
        test: Test dataset
        tokenizer: The tokenizer for formatting

    Returns:
        Tuple of formatted (train, val, test) datasets
    """
    format_fn = lambda examples: formatting_prompts_func(examples, tokenizer)

    train = train.map(format_fn, batched=True, remove_columns=train.column_names)
    val = val.map(format_fn, batched=True, remove_columns=val.column_names)
    test = test.map(format_fn, batched=True, remove_columns=test.column_names)

    return train, val, test
