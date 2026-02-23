import os
import pathlib
from dotenv import load_dotenv

# Base directory for repo
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

# Load environment from a local .env (if present)
load_dotenv()

# Also attempt to load secrets from `resources/api_keys.env` when available
alt_dotenv = BASE_DIR / "resources" / "api_keys.env"
if alt_dotenv.exists():
    load_dotenv(dotenv_path=str(alt_dotenv))

DATA_DIR = BASE_DIR / "data/processed"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DATA_PATH = DATA_DIR / "reviews_manual_1000.csv"
# DATA_PATH = DATA_DIR / "reviews_manual_sample_10.csv"

TEXT_COL = "content"  # change this if your CSV uses a different name

# ---------------------------------------------
# UPDATED LABEL SET (8 LABELS)
# ---------------------------------------------
ALLOWED_LABELS = [
    "Delivery Issue",
    "Order Accuracy",
    "App Bugs / Payment Issue",
    "Customer Support Experience",
    "Price / Cost Complaint",
    "Others"
]

LABEL_ORDER = [
    "Delivery Issue",
    "Order Accuracy",
    "App Bugs / Payment Issue",
    "Customer Support Experience",
    "Price / Cost Complaint",
    "Others"
]

# ---------------------------------------------
# 6 MODELS FROM 3+ VENDORS
# ---------------------------------------------
MODELS = [
    {"vendor": "openai", "name": "gpt-5.1"},                 # strongest OpenAI
    {"vendor": "anthropic", "name": "claude-sonnet-4-5"},    # strong Anthropic
    {"vendor": "anthropic", "name": "claude-haiku-4-5"},     # faster Anthropic
    {"vendor": "fireworks", "name": "deepseek-chat"},        # strong + cheap
    {"vendor": "openai", "name": "gpt-4.1-mini"},            # cheap OpenAI
    {"vendor": "google", "name": "gemini-2.0-flash"},        # cheap fast model
]

MODELS = [
    {"vendor": "openai", "name": "gpt-5.1"},                 # strongest OpenAI
    {"vendor": "anthropic", "name": "claude-sonnet-4-5"},    # strong Anthropic
    {"vendor": "anthropic", "name": "claude-haiku-4-5"},     # faster Anthropic
    {"vendor": "fireworks", "name": "deepseek-chat"},        # strong + cheap

]

# ---------------------------------------------
# API KEYS
# ---------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")