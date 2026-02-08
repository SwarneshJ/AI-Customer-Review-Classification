from typing import Optional

import google.generativeai as genai

from ..config import GOOGLE_API_KEY
from ..prompts import build_prompt


def init_google_client() -> Optional[object]:
    if not GOOGLE_API_KEY:
        print("Warning: GOOGLE_API_KEY not set.")
        return None
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai


def call_google(model_name: str, review: str, client=None) -> str:
    if client is None:
        raise RuntimeError("Google generative AI client is not initialized.")
    prompt = build_prompt(review)

    model = client.GenerativeModel(model_name)
    resp = model.generate_content(prompt)
    return resp.text
