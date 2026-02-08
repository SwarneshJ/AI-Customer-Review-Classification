import os
import requests

from typing import Optional

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
FIREWORKS_BASE = "https://api.fireworks.ai/v1"  # confirm with Fireworks docs

def init_deepseek_client() -> Optional[str]:
    if not FIREWORKS_API_KEY:
        print("Warning: FIREWORKS_API_KEY is not set.")
        return None
    return FIREWORKS_API_KEY

def call_deepseek(model_name: str, review: str, client=None) -> str:
    """
    Sends `review` to the DeepSeek v3 model via Fireworks.io.
    """
    if not client:
        raise RuntimeError("DeepSeek client not initialized or missing API key.")

    url = f"{FIREWORKS_BASE}/models/{model_name}/invoke"
    headers = {
        "Authorization": f"Bearer {client}",
        "Content-Type": "application/json",
    }

    data = {
        "input": review,
        "parameters": {
            "max_output_tokens": 64,
            "temperature": 0.0,
        }
    }

    resp = requests.post(url, json=data, headers=headers, timeout=30)
    resp.raise_for_status()
    out = resp.json()

    # Based on typical Fireworks API responses — adjust if necessary
    # Many Fireworks responses nest output text here:
    text = out.get("output", {}).get("text", "")
    if not text:
        # sometimes returned in a list
        text_blocks = out.get("output", {}).get("blocks", [])
        if isinstance(text_blocks, list) and text_blocks:
            text = text_blocks[0].get("text", "")
    return text.strip()
