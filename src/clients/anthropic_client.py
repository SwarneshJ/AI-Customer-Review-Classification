from typing import Optional

import anthropic

from config import ANTHROPIC_API_KEY
from prompts import build_prompt


def init_anthropic_client() -> Optional[anthropic.Anthropic]:
    if not ANTHROPIC_API_KEY:
        print("Warning: ANTHROPIC_API_KEY not set.")
        return None
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def call_anthropic(model_name: str, review: str, client=None) -> str:
    if client is None:
        raise RuntimeError("Anthropic client is not initialized.")

    prompt = build_prompt(review)

    resp = client.messages.create(
        model=model_name,
        max_tokens=64,
        temperature=0.0,
        system="You are a text classification assistant.",
        messages=[{"role": "user", "content": prompt}],
    )
    # content is a list of blocks
    return resp.content[0].text