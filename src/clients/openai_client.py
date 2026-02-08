from typing import Optional

import openai

from ..config import OPENAI_API_KEY
from ..prompts import build_prompt


def init_openai_client() -> Optional[object]:
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not set.")
        return None
    openai.api_key = OPENAI_API_KEY
    return openai


def call_openai(model_name: str, review: str, client=None) -> str:
    if client is None:
        client = openai
    if client is None:
        raise RuntimeError("OpenAI client is not initialized.")

    prompt = build_prompt(review)

    resp = client.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a text classification assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=64,
        temperature=0.0,
    )
    return resp["choices"][0]["message"]["content"]
