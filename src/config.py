import os
import pathlib
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data/processed"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DATA_PATH = DATA_DIR / "reviews_llm_15000.csv"
TEXT_COL = "review_text"  # change this if your CSV uses a different name

ALLOWED_LABELS = {
    "DELIVERY",
    "ORDER_ACCURACY",
    "FOOD_QUALITY",
    "PAYMENT",
    "APP_TECH",
    "CUSTOMER_SUPPORT",
}

LABEL_ORDER = [
    "DELIVERY",
    "ORDER_ACCURACY",
    "FOOD_QUALITY",
    "PAYMENT",
    "APP_TECH",
    "CUSTOMER_SUPPORT",
]

# 6 models from at least 3 vendors
MODELS = [
    {"vendor": "openai",    "name": "gpt-4.1-mini"},
    {"vendor": "openai",    "name": "gpt-4.1"},
    {"vendor": "anthropic", "name": "claude-3.5-sonnet"},
    {"vendor": "google",    "name": "gemini-1.5-flash"},
    {"vendor": "xai",       "name": "grok-2"},       # to fill in later
    {"vendor": "fireworks", "name": "deepseek-v3"},  # to fill in later
]

# API keys via env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")