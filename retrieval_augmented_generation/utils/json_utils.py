"""
JSON extraction and parsing utilities.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Union

from json_repair import repair_json

logger = logging.getLogger(__name__)


def extract_json_from_text(
    text: str,
    expected_keys: Optional[List[str]] = None,
    allow_arrays: bool = False
) -> Dict[str, Any]:
    """Extract and parse JSON from text response."""
    # Remove thinking tags if present
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()

    # Try to find JSON object
    start_obj = text.find("{")
    end_obj = text.rfind("}")

    # Try to find JSON array if allowed
    start_arr = text.find("[") if allow_arrays else -1
    end_arr = text.rfind("]") if allow_arrays else -1

    json_str = None
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        json_str = text[start_obj:end_obj + 1]
    elif start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        json_str = text[start_arr:end_arr + 1]

    if json_str:
        parsed = _parse_json_with_repair(json_str)

        if parsed is not None:
            if expected_keys and isinstance(parsed, dict):
                missing_keys = [key for key in expected_keys if key not in parsed]
                if missing_keys:
                    logger.warning(f"Missing expected keys: {missing_keys}")

            if isinstance(parsed, dict) and "answer" in parsed:
                return {"answer": parsed["answer"]}
            elif isinstance(parsed, (dict, list)):
                return {"answer": parsed}
            else:
                return {"answer": json_str}
        else:
            return {"answer": text.strip()}

    return {"answer": text.strip()}


def _parse_json_with_repair(json_str: str) -> Optional[Union[Dict, list]]:
    """Parse JSON with repair fallback."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    try:
        repaired_str = repair_json(json_str)
        return json.loads(repaired_str)
    except Exception:
        return None
