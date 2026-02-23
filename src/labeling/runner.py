import time
from typing import Dict

import pandas as pd


def label_dataframe_with_model(
    df: pd.DataFrame,
    text_col: str,
    vendor: str,
    model_name: str,
    call_fn,
    client=None,
    save_every: int = 100,
) -> pd.DataFrame:
    """
    For each row in df, call LLM and store the raw response as the label.
    """
    df = df.copy()
    raw_col = f"{vendor}_{model_name}_raw"
    labels_col = f"{vendor}_{model_name}_labels"

    df[raw_col] = None
    df[labels_col] = None

    n = len(df)
    print(f"Labeling {n} rows with {vendor}/{model_name}...")

    # stop after this many consecutive failures
    MAX_CONSECUTIVE_FAILURES = 3
    consecutive_failures = 0

    for i in range(n):
        review = str(df.iloc[i][text_col])

        try:
            raw = call_fn(model_name, review, client=client)
            # Directly use the raw text, stripping any accidental whitespace
            label_text = str(raw).strip() if raw else ""
            
            # success -> reset consecutive failure counter
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            # abort if too many failures in a row to avoid noisy repeated errors
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"[{vendor}/{model_name}] Fatal: {consecutive_failures} consecutive errors; aborting. Last error: {e}")
                raise
            # otherwise log and continue with empty results
            print(f"[{vendor}/{model_name}] Error on row {i}: {e} (consecutive: {consecutive_failures})")
            raw = ""
            label_text = ""

        df.at[i, raw_col] = raw
        df.at[i, labels_col] = label_text

        if (i + 1) % save_every == 0:
            print(f"[{vendor}/{model_name}] Processed {i+1}/{n} rows...")
            time.sleep(0.2)

    # remove raw response columns before returning so CSVs don't contain raw text
    raw_columns = [c for c in df.columns if c.endswith("_raw")]
    if raw_columns:
        df = df.drop(columns=raw_columns, errors="ignore")

    return df