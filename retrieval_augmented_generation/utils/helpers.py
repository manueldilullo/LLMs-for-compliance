"""
Helper utility functions for RAG system.
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_dataset(jsonl_file: str) -> List[Dict]:
    """Load Q&A dataset from JSONL file."""
    qa_pairs = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if 'question' in data and 'answer' in data:
                qa_pairs.append({
                    'question': data['question'],
                    'answer': data['answer'],
                    'refid': data.get('ref_id', ''),
                    'type': data.get('type', ''),
                    'recital_number': data.get('recital_number', ''),
                    'article_number': data.get('article_number', ''),
                    'annex_number': data.get('annex_number', '')
                })
    return qa_pairs


def sort_index_related_lists(x: List, y: List) -> Tuple[List, List]:
    """Sort two lists based on the first list's values (descending)."""
    combined = sorted(zip(x, y), key=lambda pair: pair[0], reverse=True)
    x_sorted, y_sorted = zip(*combined)
    return list(x_sorted), list(y_sorted)


def chunk_text(text: str, max_tokens: int = 128, overlap: int = 50) -> List[str]:
    """Segment text into chunks with overlap."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("BAAI/llm-embedder")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), max_tokens - overlap):
            chunk_tokens = tokens[i:i + max_tokens]
            chunks.append(tokenizer.decode(chunk_tokens))
        return chunks
    except:
        # Fallback: simple character chunking
        chunks = []
        for i in range(0, len(text), max_tokens * 4):
            chunks.append(text[i:i + max_tokens * 4])
        return [c.strip() for c in chunks if c.strip()]


def load_documents_json(json_file: str, max_tokens: int = 128) -> List[Dict[str, Any]]:
    """Load and process documents from JSON file."""
    with open(json_file, "r") as f:
        data = json.load(f)

    # Flatten AI-ACT chapters/sections into data["articles"]
    for chapter in data.get("chapters", []):
        if not data.get("articles", []):
            data["articles"] = []

        for article in chapter.get("articles", []):
            data["articles"].append({
                "number": article["number"],
                "fullText": article["fullText"],
                "relatedRecitals": article.get("relatedRecitals", []),
            })

        for section in chapter.get("sections", []):
            for article in section.get("articles", []):
                data["articles"].append({
                    "number": article["number"],
                    "fullText": article["fullText"],
                    "relatedRecitals": article.get("relatedRecitals", []),
                })

    doc_configs = [
        ("articles", "article",
         lambda x: x.get("fullText", ""),
         lambda x: {"relatedRecitals": x.get("relatedRecitals", [])}),
        ("annexes", "annex",
         lambda x: x.get("fullText", x.get("text", "")),
         lambda x: {"relatedArticles": x.get("articles", []),
                    "relatedRecitals": x.get("recitals", [])}),
        ("recitals", "recital",
         lambda x: x.get("text", ""),
         lambda x: {}),
    ]

    documents: List[Dict[str, Any]] = []

    def safe_extra(extra: Any) -> Dict[str, Any]:
        if extra is None:
            return {}
        if isinstance(extra, dict):
            return extra
        return {"_extra": extra}

    for key, doc_type, text_fn, extra_fn in doc_configs:
        if key not in data:
            continue

        for item in data[key]:
            if not isinstance(item, dict):
                continue

            text = text_fn(item)
            if not text:
                continue

            num = item.get("number", None)
            if num is None:
                continue

            parent = f"{doc_type.capitalize()}{num}"
            title = item.get("title", f"{doc_type.capitalize()} {num}")

            chunks = chunk_text(text, max_tokens=max_tokens)
            extra = safe_extra(extra_fn(item))

            for i, chunk in enumerate(chunks):
                final_text = title + " " + chunk
                doc_id = f"{doc_type.capitalize()}_{num}_chunk{i}"
                documents.append({
                    "id": doc_id,
                    "parent_id": parent,
                    "type": doc_type,
                    "number": num,
                    "text": final_text,
                    "title": title,
                    **extra,
                })

    logger.info(f"Loaded {len(documents)} chunks from {json_file}")
    return documents
