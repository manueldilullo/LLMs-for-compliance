"""
Utility helper functions for synthetic data generation.
"""

import os
import json
import asyncio
from functools import partial
from itertools import filterfalse
from typing import Dict, Any, Callable


def load_gdpr_graph(file_path: str) -> Dict[str, Any]:
    """
    Load GDPR/AI Act graph data from JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary containing the regulation data
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


async def create_llama_cpp_func(llm) -> Callable:
    """
    Create an async wrapper for llama.cpp model.

    Args:
        llm: The llama.cpp Llama instance

    Returns:
        Async callable for LLM inference
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


def truncate_dataset(
    dataset_path: str,
    output_path: str,
    keep_augmented: bool = False
) -> None:
    """
    Truncate a dataset by filtering out certain types.

    Args:
        dataset_path: Path to the full dataset
        output_path: Path to save truncated dataset
        keep_augmented: Whether to keep augmented entries
    """
    dataset = []
    with open(dataset_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                dataset.append(record)
            except json.JSONDecodeError:
                raise Exception("Cannot decode full dataset")

    if keep_augmented:
        dataset = list(filterfalse(lambda x: "augmented" not in x["type"], dataset))
    else:
        dataset = list(filterfalse(lambda x: "augmented" in x["type"], dataset))

    with open(output_path, "w", encoding='utf-8') as f:
        for record in dataset:
            f.write(json.dumps(record) + "\n")
