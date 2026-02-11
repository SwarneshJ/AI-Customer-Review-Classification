import os
import requests
from typing import Optional

from prompts import SYSTEM_PROMPT

# Expect your DeepSeek API key here:
DEEPSEEK_API_KEY = os.getenv("FIREWORKS_API_KEY")

# Official DeepSeek base URL for chat completions
DEEPSEEK_BASE = "https://api.deepseek.com/v1"


def init_deepseek_client() -> Optional[str]:
    """
    Initializes and returns the DeepSeek API key from environment.
    """
    if not DEEPSEEK_API_KEY:
        print("Warning: DEEPSEEK_API_KEY is not set.")
        return None
    return DEEPSEEK_API_KEY


def call_deepseek(model_name: str, review: str, client: Optional[str] = None) -> str:
    """
    Sends `review` to a DeepSeek chat model (e.g., deepseek-v3) using the
    official DeepSeek API.

    Parameters
    ----------
    model_name : str
        The DeepSeek model to use, e.g. "deepseek-v3".
    review : str
        The text to classify / analyze.
    client : Optional[str]
        The API key. If None, this function will raise.

    Returns
    -------
    str
        The model's response text (assistant message content).
    """
    if not client:
        raise RuntimeError("DeepSeek client not initialized or missing API key.")

    url = f"{DEEPSEEK_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {client}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": review},
        ],
        "temperature": 0.0,
        "max_tokens": 64,
    }

    resp = requests.post(url, json=data, headers=headers, timeout=30)
    resp.raise_for_status()
    out = resp.json()

    # DeepSeek uses an OpenAI-compatible response format:
    # { "choices": [ { "message": { "content": "..." } } ] }
    try:
        text = out["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        # Fallback in case of unexpected response shape
        text = str(out)

    return text.strip()
