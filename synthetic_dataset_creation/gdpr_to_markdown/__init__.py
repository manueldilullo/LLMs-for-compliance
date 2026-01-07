"""
GDPR to Markdown parser module.

This module parses GDPR markdown files and converts them to structured JSON.
"""

from .build_gdpr_json import (
    parse_article,
    parse_recital,
    parse_paragraphs,
    parse_subparagraphs,
    clean_paragraph_text,
    extract_full_text,
    build_gdpr_json,
)
