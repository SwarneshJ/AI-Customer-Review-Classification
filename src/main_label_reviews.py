import pandas as pd

from .config import (
    DATA_PATH,
    OUTPUT_DIR,
    TEXT_COL,
    MODELS,
)
from .clients.openai_client import init_openai_client, call_openai
from .clients.anthropic_client import init_anthropic_client, call_anthropic
from .clients.google_client import init_google_client, call_google
from .clients.xai_client import init_xai_client, call_xai
from .clients.fireworks_client import init_fireworks_client, call_fireworks
from .labeling.runner import label_dataframe_with_model


def get_client_and_fn(vendor: str):
    """
    Map vendor name -> (client_object, call_function).
    Extend when you implement xAI / Fireworks.
    """
    if vendor == "openai":
        client = init_openai_client()
        return client, call_openai
    elif vendor == "anthropic":
        client = init_anthropic_client()
        return client, call_anthropic
    elif vendor == "google":
        client = init_google_client()
        return client, call_google
    elif vendor == "xai":
        client = init_xai_client()
        return client, call_xai
    elif vendor == "fireworks":
        client = init_fireworks_client()
        return client, call_fireworks
    else:
        raise ValueError(f"Unknown vendor: {vendor}")


def main():
    # Load data
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if TEXT_COL not in df.columns:
        raise KeyError(
            f"Text column '{TEXT_COL}' not found. Available: {df.columns.tolist()}"
        )

    df = df.reset_index(drop=True)
    print(f"Loaded dataset with {len(df)} rows from {DATA_PATH}")

    for cfg in MODELS:
        vendor = cfg["vendor"]
        model_name = cfg["name"]

        print("\n" + "=" * 80)
        print(f"Running model: vendor={vendor}, model={model_name}")
        print("=" * 80)

        out_path = OUTPUT_DIR / f"labels_{vendor}_{model_name}.csv"
        if out_path.exists():
            print(f"Output already exists at {out_path}, skipping.")
            continue

        client, call_fn = get_client_and_fn(vendor)

        if client is None and vendor in {"openai", "anthropic", "google"}:
            print(f"Client for {vendor} not initialized, skipping this model.")
            continue

        if vendor in {"xai", "fireworks"}:
            print(f"{vendor} client not implemented yet, skipping for now.")
            continue

        labeled_df = label_dataframe_with_model(
            df=df,
            text_col=TEXT_COL,
            vendor=vendor,
            model_name=model_name,
            call_fn=call_fn,
            client=client,
            save_every=100,
        )

        labeled_df.to_csv(out_path, index=False)
        print(f"Saved labeled data for {vendor}/{model_name} to {out_path}")

    print("\nAll models finished (or skipped if not configured).")


if __name__ == "__main__":
    main()
