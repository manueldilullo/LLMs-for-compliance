"""
LLM calling and generation utilities.
"""

import asyncio
import logging
from functools import partial
from typing import Callable, Dict, Any, Optional, List, Tuple

from .json_utils import extract_json_from_text

logger = logging.getLogger(__name__)


def _join_context(docs: List[Dict[str, Any]]) -> str:
    """Join document texts into context string."""
    return "\n".join(d["text"] for d in docs if d.get("text"))


async def call_llm_with_retry(
    llm_func: Callable,
    prompt: str,
    temperature: float = 0.5,
    max_tokens: int = 512,
    retries: int = 2
) -> str:
    """Call LLM with retry logic."""
    for attempt in range(retries + 1):
        try:
            text_response = await llm_func(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["END_JSON"]
            )

            if isinstance(text_response, dict) and "choices" in text_response:
                text = text_response["choices"][0].get("text", "")
            elif isinstance(text_response, str):
                text = text_response
            else:
                text = str(text_response)

            return text

        except Exception as e:
            logger.error(f"LLM call attempt {attempt + 1} failed: {e}")
            if attempt == retries:
                raise
            continue

    raise Exception("Failed to generate response after all retries")


async def askgenerator_raw(
    llm_func: Callable,
    prompt: str,
    temperature: float = 0.5,
    max_tokens: int = 512,
    retries: int = 2,
    expected_keys: Optional[List[str]] = None,
    allow_arrays: bool = False
) -> Dict[str, Any]:
    """Generate structured response from LLM."""
    if expected_keys is None:
        expected_keys = ["answer"]

    try:
        text = await call_llm_with_retry(
            llm_func=llm_func,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries
        )
        return extract_json_from_text(
            text=text,
            expected_keys=expected_keys,
            allow_arrays=allow_arrays
        )

    except Exception as e:
        logger.error(f"askgenerator_raw failed: {e}")
        return {"answer": f"Error: {str(e)}"}


async def init_llm(model_name: str, model_dic: Dict[str, Any]) -> Tuple[Callable, Any]:
    """
    Initialize LLM from HuggingFace and create async wrapper.
    
    Args:
        model_name: Name of the model to load (key in model_dic)
        model_dic: Dictionary containing model configurations with keys:
            - HF_REPO_NAME: HuggingFace repository ID
            - HF_MODEL_NAME: Model filename
            
    Returns:
        tuple: (llm_func, llm) - async callable and raw llm instance
        
    Example:
        >>> model_dic = {
        ...     "Llama-3.2-3B": {
        ...         "HF_REPO_NAME": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        ...         "HF_MODEL_NAME": "Llama-3.2-3B-Instruct-f16.gguf"
        ...     }
        ... }
        >>> llm_func, llm = await init_llm("Llama-3.2-3B", model_dic)
    """
    try:
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        raise ImportError(
            "Please install llama-cpp-python and huggingface-hub: "
            "pip install llama-cpp-python huggingface-hub"
        )
    
    if model_name not in model_dic:
        raise ValueError(
            f"Model '{model_name}' not found in model_dic. "
            f"Available models: {list(model_dic.keys())}"
        )
    
    model_config = model_dic[model_name]
    HF_REPO_NAME = model_config["HF_REPO_NAME"]
    HF_MODEL_NAME = model_config["HF_MODEL_NAME"]
    
    logger.info(f"Downloading model {model_name} from HuggingFace...")
    logger.info(f"  Repository: {HF_REPO_NAME}")
    logger.info(f"  File: {HF_MODEL_NAME}")
    
    # Download model from HuggingFace Hub
    model_path = hf_hub_download(
        repo_id=HF_REPO_NAME,
        filename=HF_MODEL_NAME,
        local_dir="./models"
    )
    
    logger.info(f"Model downloaded to: {model_path}")
    logger.info("Initializing Llama model...")
    
    # Initialize Llama model
    llm = Llama(
        model_path=model_path,
        n_threads=2,
        n_batch=512,
        n_gpu_layers=40,  # Adjust based on your GPU VRAM
        n_ctx=4096,
        verbose=False,
    )
    
    logger.info("Creating async wrapper for LLM...")
    
    # Create async wrapper function
    async def llm_func(prompt: str, stop: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Async wrapper for llama.cpp model.
        
        Args:
            prompt: The prompt to generate from
            stop: List of stop sequences
            **kwargs: Additional parameters (max_tokens, temperature, etc.)
            
        Returns:
            Response dictionary from llama.cpp
        """
        loop = asyncio.get_running_loop()
        
        # Prepare parameters
        params = {
            "prompt": prompt,
            "stop": stop or [],
            "max_tokens": kwargs.pop("max_tokens", 512),
            "temperature": kwargs.pop("temperature", 0.3),
            **kwargs
        }
        
        # Run in thread pool executor
        call_func = partial(llm, **params)
        response = await loop.run_in_executor(None, call_func)
        
        return response
    
    logger.info(f"LLM {model_name} initialized successfully!")
    
    return llm_func, llm
