from typing import Optional

import google.generativeai as genai

from config import GOOGLE_API_KEY
from prompts import build_prompt


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
    # Let errors from the client propagate so the caller can log them
    model = client.GenerativeModel(model_name)
    resp = model.generate_content(prompt)

    # Robust extraction of text from various response structures
    try:
        if resp is None:
            return ""

        if hasattr(resp, "text") and resp.text:
            return resp.text

        candidates = getattr(resp, "candidates", None)
        if candidates:
            first = candidates[0]
            if isinstance(first, dict):
                for key in ("content", "text", "output"):
                    if key in first and first[key]:
                        return first[key]
            else:
                for key in ("content", "text", "output"):
                    if hasattr(first, key):
                        val = getattr(first, key)
                        if isinstance(val, str) and val:
                            return val

        output = getattr(resp, "output", None)
        if output:
            if isinstance(output, list) and len(output) > 0:
                first = output[0]
                if isinstance(first, dict) and "content" in first:
                    return first["content"]

        return str(resp)
    except Exception:
        return ""
