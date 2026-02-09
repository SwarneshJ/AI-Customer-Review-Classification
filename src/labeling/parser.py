import json
from typing import List

from config import ALLOWED_LABELS, LABEL_ORDER


def parse_labels(raw: str) -> List[str]:
    """
    Parse the model output into a list of valid labels in a stable order.
    Returns [] on failure.
    """
    if raw is None:
        return []
    raw = str(raw).strip()

    try:
        # If it's not starting with '[', try to slice the JSON array out of text
        if not raw.startswith("["):
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1:
                raw = raw[start : end + 1]

        data = json.loads(raw)
        if not isinstance(data, list):
            return []

        cleaned = []
        for item in data:
            if not isinstance(item, str):
                continue
            lab = item.strip().upper()
            if lab in ALLOWED_LABELS:
                cleaned.append(lab)

        # preserve canonical order
        ordered = [lab for lab in LABEL_ORDER if lab in cleaned]
        return ordered

    except Exception:
        return []