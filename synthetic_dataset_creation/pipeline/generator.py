"""
Asynchronous Q&A generation pipeline for legal documents.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Callable, Type, Optional, Set

from pydantic import BaseModel, ValidationError
from json_repair import repair_json

from ..config import UnityConfig, BindingConfig
from ..schemas import (
    ArticleUnityQuestion,
    ArticleUnityAnswer,
    RecitalUnityQuestion,
    RecitalUnityAnswer,
    AnnexUnityQuestion,
    AnnexUnityAnswer,
    BindingArticleRecitalQuestion,
    BindingArticleRecitalAnswer,
    BindingAnnexArticleQuestion,
    BindingAnnexArticleAnswer,
    BindingAnnexRecitalQuestion,
    BindingAnnexRecitalAnswer,
    AugmentedVariants,
)


class AsyncQAGDPRPipeline:
    """
    Asynchronous pipeline for generating Q&A pairs from legal documents.
    
    This pipeline supports:
    - Unity generation (single article/recital/annex)
    - Binding generation (relationships between entities)
    - Augmentation of existing Q&A pairs
    - Resume from checkpoint
    """

    def __init__(
        self,
        data: Dict[str, Any],
        llm_func: Callable,
        output_path: str = "dataset.jsonl",
        max_concurrency: int = 1,
        prompts: Dict[str, str] = None
    ):
        """
        Initialize the pipeline.

        Args:
            data: Dictionary containing articles, recitals, and annexes
            llm_func: Async callable for LLM inference
            output_path: Path to save generated Q&A pairs
            max_concurrency: Maximum concurrent LLM calls
            prompts: Dictionary of prompt templates
        """
        self.data = data
        self.dataset: List[Dict[str, Any]] = []
        self.llm_func = llm_func
        self.output_path = output_path
        self.sem = asyncio.Semaphore(max_concurrency)
        self.write_lock = asyncio.Lock()
        self.prompts = prompts or {}
        self.processed_ids: Set[str] = self._load_progress()
        print(f"[INIT] Pipeline initialized. Resuming with {len(self.processed_ids)} items.")

    def _load_progress(self) -> Set[str]:
        """Load previously processed IDs for resume capability."""
        ids = set()
        if not os.path.exists(self.output_path):
            return ids
        try:
            with open(self.output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        self.dataset.append(record)
                        unique_key = f"{record.get('type')}|{record.get('ref_id')}"
                        ids.add(unique_key)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[WARN] Error reading file: {e}")
        return ids

    async def _save_record(self, record: Dict[str, Any]) -> None:
        """Save a single record to the output file."""
        unique_key = f"{record.get('type')}|{record.get('ref_id')}"
        if unique_key in self.processed_ids:
            return

        try:
            self.dataset.append(record)
            async with self.write_lock:
                with open(self.output_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record) + "\n")
                self.processed_ids.add(unique_key)
        except Exception as e:
            print(f"[ERROR] Save failed: {e}")

    async def _generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        retries: int = 2
    ) -> Optional[BaseModel]:
        """Generate structured output from LLM with JSON parsing."""
        async with self.sem:
            for attempt in range(retries + 1):
                try:
                    text_response = await self.llm_func(
                        prompt=prompt,
                        temperature=0.7,
                        max_tokens=512,
                        stop=["END_JSON"]
                    )

                    start = text_response.find("{")
                    if start != -1:
                        json_str = text_response[start:]
                        if not json_str.strip().endswith("}"):
                            json_str += "}"
                    else:
                        json_str = text_response

                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError:
                        data = json.loads(repair_json(json_str))

                    return schema(**data)

                except (ValidationError, Exception) as e:
                    if attempt < retries:
                        continue
                    print(f"[FAIL] {e}")
                    return None
        return None

    async def _generate_unity(self, config: UnityConfig, limit: int = None, n: int = 5) -> None:
        """Generic method to generate unity-type Q&A for articles, recitals, or annexes."""
        print(f"--- {config.entity_name} Unity ---")

        items = self.data.get(config.data_key, [])
        if limit:
            items = items[:limit]

        tasks = []
        for item in items:
            item_num = item['number']
            if f"{config.type_name}|{item_num}" in self.processed_ids:
                continue

            tasks.append(self._process_unity_item(config, item))

            sub_items = item.get(config.sub_items_key, [])
            for idx, sub_item in enumerate(sub_items):
                sub_text = sub_item.get('text', sub_item) if isinstance(sub_item, dict) else sub_item
                tasks.append(self._process_unity_item(config, {
                    'number': item_num,
                    config.text_key: sub_text,
                    config.sub_item_num_key: idx
                }))

        assert n > 0
        if tasks:
            await asyncio.gather(*(tasks * n))

    async def _process_unity_item(self, config: UnityConfig, item: Dict[str, Any]) -> None:
        """Process a single unity item."""
        item_num = item['number']
        sub_num = item.get(config.sub_item_num_key, 0)
        text = item.get(config.text_key, '')

        q_prompt = config.q_format_fn(self.prompts[config.question_prompt_key], item_num, text)
        q_res = await self._generate_structured(q_prompt, config.question_schema)
        if not q_res:
            return

        a_prompt = config.a_format_fn(self.prompts[config.answer_prompt_key], item_num, text)
        a_res = await self._generate_structured(a_prompt, config.answer_schema)
        if not a_res:
            return

        ref_id = f"{item_num}_{sub_num}" if sub_num else item_num
        await self._save_record({
            "type": config.type_name,
            "ref_id": ref_id,
            "question": q_res.question,
            "answer": a_res.answer
        })
        print(f"[OK] {config.entity_name} {item_num}")

    async def _generate_binding(self, config: BindingConfig, limit: int = None, n: int = 5) -> None:
        """Generic method to generate binding Q&A between two entity types."""
        print(f"--- Binding {config.primary_name} - {config.secondary_name} Questions ---")

        secondary_items = self.data.get(config.secondary_data_key, [])
        secondary_lookup = {
            item['number']: item.get(config.secondary_text_key, '')
            for item in secondary_items
        }

        primary_items = self.data.get(config.primary_data_key, [])
        if limit:
            primary_items = primary_items[:limit]

        tasks = []
        for primary in primary_items:
            primary_num = primary['number']
            primary_text = primary.get(config.primary_text_key, '')

            for secondary_num in primary.get(config.relation_key, []):
                unique_key = f"{config.type_name}|{primary_num}_{secondary_num}"
                if unique_key in self.processed_ids:
                    continue

                secondary_text = secondary_lookup.get(secondary_num)
                if not secondary_text:
                    continue

                tasks.append(self._process_binding_item(
                    config, primary_num, primary_text, secondary_num, secondary_text
                ))

        assert n > 0
        if tasks:
            await asyncio.gather(*(tasks * n))

    async def _process_binding_item(
        self,
        config: BindingConfig,
        primary_num: int,
        primary_text: str,
        secondary_num: int,
        secondary_text: str
    ) -> None:
        """Process a single binding item."""
        q_prompt = config.q_format_fn(
            self.prompts[config.question_prompt_key],
            primary_num, primary_text, secondary_num, secondary_text
        )
        q_res = await self._generate_structured(q_prompt, config.question_schema)
        if not q_res:
            return

        a_prompt = config.a_format_fn(
            self.prompts[config.answer_prompt_key],
            primary_num, primary_text, secondary_num, secondary_text
        )
        a_res = await self._generate_structured(a_prompt, config.answer_schema)
        if not a_res:
            return

        await self._save_record({
            "type": config.type_name,
            "ref_id": f"{primary_num}_{secondary_num}",
            config.primary_name.lower(): primary_num,
            config.secondary_name.lower(): secondary_num,
            "question": q_res.question,
            "answer": a_res.answer
        })
        print(f"[OK] Binding {primary_num}-{secondary_num}")

    # =========================================================================
    # Public API Methods
    # =========================================================================

    async def generate_article_unity(self, limit: int = None, n: int = 5) -> None:
        """Generate Q&A pairs for individual articles."""
        config = UnityConfig(
            data_key='articles',
            type_name='article_unity',
            entity_name='Article',
            text_key='fullText',
            sub_items_key='paragraphs',
            sub_item_num_key='paragraph_num',
            question_schema=ArticleUnityQuestion,
            answer_schema=ArticleUnityAnswer,
            question_prompt_key='PROMPT_ARTICLE_UNITY_Q',
            answer_prompt_key='PROMPT_ARTICLE_UNITY_A',
            q_format_fn=lambda p, num, txt: p.format(number=num, metadata="", full_text=txt, excerpt=txt),
            a_format_fn=lambda p, num, txt: p.format(number=num, full_text=txt, excerpt=txt),
        )
        await self._generate_unity(config, limit, n)

    async def generate_recital_unity(self, limit: int = None, n: int = 5) -> None:
        """Generate Q&A pairs for individual recitals."""
        config = UnityConfig(
            data_key='recitals',
            type_name='recital_unity',
            entity_name='Recital',
            text_key='text',
            sub_items_key='sentences',
            sub_item_num_key='sentence_num',
            question_schema=RecitalUnityQuestion,
            answer_schema=RecitalUnityAnswer,
            question_prompt_key='PROMPT_RECITAL_UNITY_Q',
            answer_prompt_key='PROMPT_RECITAL_UNITY_A',
            q_format_fn=lambda p, num, txt: p.format(recital_number=num, recital_text=txt),
            a_format_fn=lambda p, num, txt: p.format(recital_number=num, recital_text=txt, answer_excerpt=txt),
        )
        await self._generate_unity(config, limit, n)

    async def generate_annex_unity(self, limit: int = None, n: int = 5) -> None:
        """Generate Q&A pairs for individual annexes."""
        config = UnityConfig(
            data_key='annexes',
            type_name='annex_unity',
            entity_name='Annex',
            text_key='fullText',
            sub_items_key='paragraphs',
            sub_item_num_key='paragraph_num',
            question_schema=AnnexUnityQuestion,
            answer_schema=AnnexUnityAnswer,
            question_prompt_key='PROMPT_ANNEX_UNITY_Q',
            answer_prompt_key='PROMPT_ANNEX_UNITY_A',
            q_format_fn=lambda p, num, txt: p.format(number=num, metadata="", full_text=txt, excerpt=txt),
            a_format_fn=lambda p, num, txt: p.format(number=num, full_text=txt, excerpt=txt),
        )
        await self._generate_unity(config, limit, n)

    async def generate_binding_article_recital_questions(self, limit: int = None, n: int = 5) -> None:
        """Generate Q&A pairs for article-recital relationships."""
        config = BindingConfig(
            primary_data_key='articles',
            secondary_data_key='recitals',
            relation_key='relatedRecitals',
            type_name='binding_question_article_recital',
            primary_name='Article',
            secondary_name='Recital',
            primary_text_key='fullText',
            secondary_text_key='text',
            question_schema=BindingArticleRecitalQuestion,
            answer_schema=BindingArticleRecitalAnswer,
            question_prompt_key='PROMPT_ARTICLE_RECITAL_BINDING_Q',
            answer_prompt_key='PROMPT_ARTICLE_RECITAL_BINDING_A',
            q_format_fn=lambda p, art, art_txt, rec, rec_txt: p.format(
                recital_num=rec, article_num=art, metadata="",
                article_text=art_txt, recital_text=rec_txt
            ),
            a_format_fn=lambda p, art, art_txt, rec, rec_txt: p.format(
                recital_number=rec, article_number=art,
                article_text=art_txt, recital_text=rec_txt, answer_excerpt=rec_txt
            ),
        )
        await self._generate_binding(config, limit, n)

    async def generate_binding_annex_article_questions(self, limit: int = None, n: int = 5) -> None:
        """Generate Q&A pairs for annex-article relationships."""
        config = BindingConfig(
            primary_data_key='annexes',
            secondary_data_key='articles',
            relation_key='articles',
            type_name='binding_question_annex_article',
            primary_name='Annex',
            secondary_name='Article',
            primary_text_key='fullText',
            secondary_text_key='fullText',
            question_schema=BindingAnnexArticleQuestion,
            answer_schema=BindingAnnexArticleAnswer,
            question_prompt_key='PROMPT_ANNEX_ARTICLE_BINDING_Q',
            answer_prompt_key='PROMPT_ANNEX_ARTICLE_BINDING_A',
            q_format_fn=lambda p, ann, ann_txt, art, art_txt: p.format(
                annex_num=ann, article_num=art, metadata="",
                article_text=art_txt, annex_text=ann_txt
            ),
            a_format_fn=lambda p, ann, ann_txt, art, art_txt: p.format(
                annex_number=ann, article_number=art,
                article_text=art_txt, annex_text=ann_txt, answer_excerpt=ann_txt
            ),
        )
        await self._generate_binding(config, limit, n)

    async def generate_binding_annex_recital_questions(self, limit: int = None, n: int = 5) -> None:
        """Generate Q&A pairs for annex-recital relationships."""
        config = BindingConfig(
            primary_data_key='annexes',
            secondary_data_key='recitals',
            relation_key='recitals',
            type_name='binding_question_annex_recital',
            primary_name='Annex',
            secondary_name='Recital',
            primary_text_key='fullText',
            secondary_text_key='text',
            question_schema=BindingAnnexRecitalQuestion,
            answer_schema=BindingAnnexRecitalAnswer,
            question_prompt_key='PROMPT_ANNEX_RECITAL_BINDING_Q',
            answer_prompt_key='PROMPT_ANNEX_RECITAL_BINDING_A',
            q_format_fn=lambda p, ann, ann_txt, rec, rec_txt: p.format(
                annex_num=ann, recital_num=rec, metadata="",
                recital_text=rec_txt, annex_text=ann_txt
            ),
            a_format_fn=lambda p, ann, ann_txt, rec, rec_txt: p.format(
                annex_number=ann, recital_number=rec,
                recital_text=rec_txt, annex_text=ann_txt, answer_excerpt=rec_txt
            ),
        )
        await self._generate_binding(config, limit, n)

    async def augment_dataset(self, n: int = 5) -> None:
        """Augment existing Q&A pairs with question variations."""
        print("--- Augmentation ---")
        if not os.path.exists(self.output_path):
            return

        with open(self.output_path, 'r') as f:
            lines = f.readlines()

        tasks = []
        for line in lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "augmented" in item['type']:
                continue

            if f"{item['type']}_augmented|{item['ref_id']}_aug_0" in self.processed_ids:
                continue

            tasks.append(self._process_augmentation(item, n))

        if tasks:
            await asyncio.gather(*tasks)

    async def _process_augmentation(self, item: Dict[str, Any], n: int = 5) -> None:
        """Process augmentation for a single Q&A pair."""
        prompt = self.prompts['PROMPT_AUGMENTATION_Q'].format(
            original_question=item['question'],
            full_context="",
            original_answer=item['answer'],
            n=n
        )
        variants = await self._generate_structured(prompt, AugmentedVariants)

        if variants:
            for i, q_var in enumerate(variants.questions):
                await self._save_record({
                    "type": f"{item['type']}_augmented",
                    "ref_id": f"{item['ref_id']}_aug_{i}",
                    "question": q_var,
                    "answer": item['answer'],
                    "original_ref_type": item['type']
                })
            print(f"[OK] Augmented {item['ref_id']}")
