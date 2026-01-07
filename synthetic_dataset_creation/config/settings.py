"""
Configuration dataclasses for synthetic data generation pipeline.
"""

from dataclasses import dataclass
from typing import Type, Callable
from pydantic import BaseModel


@dataclass
class UnityConfig:
    """
    Configuration for unity-type generation (article, recital, annex).
    
    Unity generation creates Q&A pairs for individual legal entities.
    """
    data_key: str
    type_name: str
    entity_name: str
    text_key: str
    sub_items_key: str
    sub_item_num_key: str
    question_schema: Type[BaseModel]
    answer_schema: Type[BaseModel]
    question_prompt_key: str
    answer_prompt_key: str
    q_format_fn: Callable
    a_format_fn: Callable


@dataclass
class BindingConfig:
    """
    Configuration for binding-type generation (article-recital, annex-article, etc.).
    
    Binding generation creates Q&A pairs that capture relationships between
    two different legal entities.
    """
    primary_data_key: str
    secondary_data_key: str
    relation_key: str
    type_name: str
    primary_name: str
    secondary_name: str
    primary_text_key: str
    secondary_text_key: str
    question_schema: Type[BaseModel]
    answer_schema: Type[BaseModel]
    question_prompt_key: str
    answer_prompt_key: str
    q_format_fn: Callable
    a_format_fn: Callable
