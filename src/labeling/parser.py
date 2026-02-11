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

        # Build a normalized lookup for allowed labels (normalize removes
        # non-alphanumeric characters and lowercases)
        def normalize(s: str) -> str:
            return "".join(ch for ch in s.lower() if ch.isalnum())

        allowed_map = {normalize(lbl): lbl for lbl in LABEL_ORDER}

        cleaned = []
        for item in data:
            if not isinstance(item, str):
                continue
            norm_item = normalize(item.strip())
            if norm_item in allowed_map:
                cleaned.append(allowed_map[norm_item])
                continue

            # as a fallback, check if any allowed label normalized value
            # is contained within the normalized item (handles small deviations)
            for key, orig in allowed_map.items():
                if key in norm_item:
                    cleaned.append(orig)
                    break

        # preserve configured order
        ordered = [lab for lab in LABEL_ORDER if lab in cleaned]
        return ordered

    except Exception:
        return []