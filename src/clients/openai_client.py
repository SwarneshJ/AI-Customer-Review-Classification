from openai import OpenAI
from config import OPENAI_API_KEY
from prompts import SYSTEM_PROMPT, build_prompt

def init_openai_client():
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY is not set.")
        return None
    client = OpenAI(api_key=OPENAI_API_KEY)
    return client

def call_openai(model_name: str, review: str, client=None) -> str:
    if client is None:
        raise RuntimeError("OpenAI client not initialized.")

    prompt = build_prompt(review)

    response = client.chat.completions.create(
        model=model_name,  # e.g., "gpt-5.1" or "gpt-4.1-mini"
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=64,
    )

    return response.choices[0].message.content.strip()
