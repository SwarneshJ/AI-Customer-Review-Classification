import json
import time
from typing import Dict

import pandas as pd

from ..labeling.parser import parse_labels


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
    For each row in df, call LLM and store:
      - raw response
      - parsed_labels (JSON string)
    """
    df = df.copy()
    raw_col = f"{vendor}_{model_name}_raw"
    labels_col = f"{vendor}_{model_name}_labels"

    df[raw_col] = None
    df[labels_col] = None

    n = len(df)
    print(f"Labeling {n} rows with {vendor}/{model_name}...")

    for i in range(n):
        review = str(df.iloc[i][text_col])

        try:
            raw = call_fn(model_name, review, client=client)
            labels = parse_labels(raw)
        except Exception as e:
            print(f"[{vendor}/{model_name}] Error on row {i}: {e}")
            raw = ""
            labels = []

        df.at[i, raw_col] = raw
        df.at[i, labels_col] = json.dumps(labels)

        if (i + 1) % save_every == 0:
            print(f"[{vendor}/{model_name}] Processed {i+1}/{n} rows...")
            time.sleep(0.2)

    return df
