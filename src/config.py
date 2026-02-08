import os
import pathlib
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data/processed"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# DATA_PATH = DATA_DIR / "reviews_manual_1000.csv"
DATA_PATH = DATA_DIR / "reviews_manual_sample_10.csv"

TEXT_COL = "review_text"  # change this if your CSV uses a different name

# ---------------------------------------------
# UPDATED LABEL SET (8 LABELS)
# ---------------------------------------------
ALLOWED_LABELS = {
    "DELIVERY",
    "ORDER_ACCURACY",
    "FOOD_QUALITY",
    "PAYMENT",
    "APP_TECH",
    "CUSTOMER_SUPPORT",
    "PRICE_COST",
    "OTHERS",
}

LABEL_ORDER = [
    "DELIVERY",
    "ORDER_ACCURACY",
    "FOOD_QUALITY",
    "PAYMENT",
    "APP_TECH",
    "CUSTOMER_SUPPORT",
    "PRICE_COST",
    "OTHERS",
]

# ---------------------------------------------
# 6 MODELS FROM 3+ VENDORS
# ---------------------------------------------
MODELS = [
    {"vendor": "openai", "name": "gpt-5.1"},                 # strongest OpenAI
    {"vendor": "anthropic", "name": "claude-3.5-sonnet"},    # strong Anthropic
    {"vendor": "google", "name": "gemini-pro"},              # strong Google
    {"vendor": "fireworks", "name": "deepseek-v3"},          # strong + cheap
    {"vendor": "xai", "name": "grok-2"},                     # independent vendor
    {"vendor": "google", "name": "gemini-flash"},            # cheap fast model
]

# ---------------------------------------------
# API KEYS
# ---------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")