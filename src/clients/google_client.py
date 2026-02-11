from typing import Optional
import re
import time
import random

import google.generativeai as genai
from google.api_core.exceptions import (
    ResourceExhausted,
    InternalServerError,
    ServiceUnavailable,
    DeadlineExceeded,
)

from config import GOOGLE_API_KEY
from prompts import build_prompt


def init_google_client() -> Optional[object]:
    if not GOOGLE_API_KEY:
        print("Warning: GOOGLE_API_KEY not set.")
        return None
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai


def _extract_gemini_text(resp) -> str:
    """
    Robustly extract plain text from a Gemini response object WITHOUT EVER
    TOUCHING resp.text (because that can raise when finish_reason != normal).
    """
    if resp is None:
        return ""

    # Try candidates[0].content.parts[*].text
    candidates = getattr(resp, "candidates", None)
    if candidates:
        first = candidates[0]
        content = getattr(first, "content", None)
        parts = getattr(content, "parts", None) if content else None

        if parts:
            texts = []
            for p in parts:
                t = getattr(p, "text", None)
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
            if texts:
                return "\n".join(texts)

        # No parts at all → usually safety block
        finish_reason = getattr(first, "finish_reason", None)
        return f"[SAFETY_BLOCK finish_reason={finish_reason}]"

    # No candidates → just fall back to string for debugging
    return str(resp)


# Relax safety as much as the API allows (Gemini still has core safety you can't turn off)
DEFAULT_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]


def call_google(
    model_name: str,
    review: str,
    client=None,
    max_retries: int = 5,
    base_backoff: float = 1.0,
) -> str:
    """
    Call a Gemini model (e.g. 'gemini-2.0-flash' or 'gemini-2.5-pro') with
    retry + backoff to handle 429 (Resource exhausted) and transient server errors.

    Returns a string label (or raw model output) and NEVER crashes on SDK's
    `response.text` accessor.
    """
    if client is None:
        raise RuntimeError("Google generative AI client is not initialized.")

    prompt = build_prompt(review)

    model = client.GenerativeModel(
        model_name,
        safety_settings=DEFAULT_SAFETY_SETTINGS,
    )

    last_error = None

    for attempt in range(max_retries):
        try:
            resp = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": 64,
                },
            )

            out = _extract_gemini_text(resp)

            # If safety-blocked, treat as OTHER so the pipeline keeps going
            if out.startswith("[SAFETY_BLOCK"):
                return "OTHER"

            # Detect obvious quota/rate-limit messages in the text
            ERROR_PATTERNS = re.compile(
                r"(?i)(quota|exceed|rate limit|429|rate-limit|quota exceeded|quota_exceeded)"
            )
            if out and ERROR_PATTERNS.search(out):
                first_line = out.splitlines()[0]
                snippet = first_line[:300]
                raise RuntimeError(f"Google API error detected: {snippet}")

            if not out.strip():
                # Try to surface more info if available
                pf = getattr(resp, "prompt_feedback", None)
                if pf and getattr(pf, "block_reason", None):
                    # Still just map to OTHER for labeling
                    return "OTHER"

                cands = getattr(resp, "candidates", None)
                if cands:
                    fr = getattr(cands[0], "finish_reason", None)
                    # Again, map to OTHER so you don't hard-fail your run
                    return "OTHER"

                # Truly empty / weird response, but don't crash the entire batch
                return "OTHER"

            return out.strip()

        except (ResourceExhausted, InternalServerError, ServiceUnavailable, DeadlineExceeded) as e:
            last_error = e
            if attempt == max_retries - 1:
                raise RuntimeError(f"Google API transient error after retries: {e}") from e

            sleep_for = base_backoff * (2 ** attempt) + random.uniform(0, 0.5)
            print(
                f"[Google/Gemini] Transient error ({type(e).__name__}): {e}. "
                f"Retrying in {sleep_for:.1f}s (attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(sleep_for)

        except Exception as e:
            # This bubbles up to your runner, which stops after 3 consecutive failures
            raise

    if last_error:
        raise RuntimeError(f"Google API failed after retries: {last_error}")
    raise RuntimeError("Google API failed for unknown reasons.")
