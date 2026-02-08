import os
import requests

from typing import Optional

XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_BASE_URL = "https://api.x.ai/v1"  # your endpoint may vary

def init_grok_client() -> Optional[str]:
    if not XAI_API_KEY:
        print("Warning: XAI_API_KEY is not set.")
        return None
    return XAI_API_KEY

def call_grok(model_name: str, review: str, client=None) -> str:
    """
    Sends `review` to the xAI Grok model.
    """
    if not client:
        raise RuntimeError("Grok client not initialized or missing API key.")

    url = f"{XAI_BASE_URL}/models/{model_name}/generate"
    headers = {
        "Authorization": f"Bearer {client}",
        "Content-Type": "application/json",
    }

    payload = {
        "input": review,
        "parameters": {
            "max_completion_tokens": 64,
            "temperature": 0.0,
        },
    }

    response = requests.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    out = response.json()

    # Most xAI endpoints put text here:
    if "text" in out:
        return out["text"].strip()
    # If API returns blocks, adjust accordingly:
    blocks = out.get("output", {}).get("blocks", [])
    if isinstance(blocks, list) and blocks:
        return blocks[0].get("text", "").strip()
    return ""
