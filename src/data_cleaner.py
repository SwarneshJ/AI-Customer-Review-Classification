import pandas as pd
import re
from unidecode import unidecode   # pip install unidecode
from langdetect import detect, LangDetectException   # pip install langdetect

# 1) Load data
df = pd.read_excel("food_delivery_apps.xlsx")   # or pd.read_csv("file.csv")

# Drop rows with no review text
df = df[df["content"].notna()].copy()

# 2) Text cleaning (in-place on 'content')
MOJIBAKE_REPLACEMENTS = {
    "‚Äô": "'",   # don‚Äôt -> don't
    "‚Äö": "'",   # I‚Äôm -> I'm
    "‚Äú": '"',
    "‚Äù": '"',
    "√§": "a",
    "√©": "e",
    "√®": "i",
    "√±": "n",
    "√≥": ">=",
    "√≤": "<=",
}

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    
    # 1) fix common mojibake sequences
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        s = s.replace(bad, good)
    
    # 2) normalize accents / fancy punctuation
    s = unidecode(s)
    
    # 3) normalize whitespace and line breaks
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    
    # 4) strip outer spaces
    return s.strip()

# overwrite raw content directly
df["content"] = df["content"].apply(clean_text)

# 3) Filter very short reviews
df = df[df["content"].str.len() >= 20]

# 4) Language detection and filter to English
def safe_lang_detect(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

df["lang"] = df["content"].apply(safe_lang_detect)
df = df[df["lang"] == "en"].copy()

# 5) Drop duplicate users (one review per userName)
df = df.drop_duplicates(subset=["userName"])

# 6) Drop unimportant columns
cols_to_drop = ["replyContent", "repliedDate", "appVersion", "content_clean", "lang"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# 7) Create manual (1k) and LLM (15k) splits
df_manual = df.sample(n=1000, random_state=42)
df_llm = df.drop(df_manual.index).sample(n=15000, random_state=42)

# 8) Save
df_manual.to_csv("reviews_manual_1000.csv", index=False)
df_llm.to_csv("reviews_llm_15000.csv", index=False)
