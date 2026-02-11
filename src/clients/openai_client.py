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

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # Choose token parameter based on model name (gpt-5.x uses `max_completion_tokens`)
    token_kwargs = {}
    if model_name and model_name.startswith("gpt-5"):
        token_kwargs["max_completion_tokens"] = 64
    else:
        token_kwargs["max_tokens"] = 64

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            **token_kwargs,
        )
    except Exception as e:
        # If the model rejected the chosen token parameter, try the other one.
        err = str(e)
        if "max_tokens" in err and "not supported" in err or "Unsupported parameter" in err:
            # swap to the other parameter and retry
            alt_kwargs = {}
            if "max_tokens" in token_kwargs:
                alt_kwargs["max_completion_tokens"] = token_kwargs.get("max_tokens", 64)
            else:
                alt_kwargs["max_tokens"] = token_kwargs.get("max_completion_tokens", 64)

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                **alt_kwargs,
            )
        else:
            raise

    return response.choices[0].message.content.strip()
