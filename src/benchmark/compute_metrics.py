import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, matthews_corrcoef
import krippendorff


# ---------------------------
# CONFIG
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent

GOLD_PATH = BASE_DIR / "data" / "processed" / "reviews_manual_1000_gold.csv"
MODEL_DIR = BASE_DIR / "outputs"
OUT_DIR = BASE_DIR / "outputs"

OUT_DIR.mkdir(parents=True, exist_ok=True)

GOLD_COL = "gold_label"

ALLOWED_LABELS = [
    "Delivery Issue",
    "Order Accuracy",
    "App Bugs / Payment Issue",
    "Customer Support Experience",
    "Price / Cost Complaint",
    "Others"
]


# ---------------------------
# Helpers
# ---------------------------
def clean_label(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "Others"
    return str(x).strip()


def parse_model_labels_cell(cell):
    """
    Model outputs look like:
        ["Delivery Issues"]
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return "Others"

    s = str(cell).strip()

    if not s:
        return "Others"

    if s.startswith("["):
        try:
            arr = json.loads(s)
            if isinstance(arr, list) and len(arr) > 0:
                return clean_label(arr[0])
            return "Others"
        except Exception:
            return "Others"

    return clean_label(s)


def find_pred_col(df):
    label_cols = [c for c in df.columns if c.endswith("_labels")]
    if label_cols:
        return label_cols[0]

    raise ValueError(f"Prediction column not found in {list(df.columns)}")


# ---------------------------
# MAIN
# ---------------------------
def main():

    print("Loading gold file...")
    gold = pd.read_csv(GOLD_PATH)

    if GOLD_COL not in gold.columns:
        raise ValueError(f"Gold file must contain column '{GOLD_COL}'")

    gold["gold_clean"] = gold[GOLD_COL].apply(clean_label)

    model_files = sorted(MODEL_DIR.glob("labels_*.csv"))

    if not model_files:
        raise FileNotFoundError("No model output files found.")

    summary_rows = []
    per_class_rows = []

    for mf in model_files:

        print(f"\nEvaluating {mf.name}")
        model_df = pd.read_csv(mf)

        if len(gold) != len(model_df):
            raise ValueError(
                f"Row mismatch: gold={len(gold)} vs model={len(model_df)}"
            )

        pred_col = find_pred_col(model_df)
        model_df["pred_clean"] = model_df[pred_col].apply(parse_model_labels_cell)

        y_true = gold["gold_clean"]
        y_pred = model_df["pred_clean"]

        report = classification_report(
            y_true,
            y_pred,
            labels=ALLOWED_LABELS,
            output_dict=True,
            zero_division=0
        )

        macro_f1 = report["macro avg"]["f1-score"]
        weighted_f1 = report["weighted avg"]["f1-score"]
        mcc = matthews_corrcoef(y_true, y_pred)

        alpha = krippendorff.alpha(
            reliability_data=[y_true.tolist(), y_pred.tolist()],
            level_of_measurement="nominal"
        )

        model_tag = mf.stem.replace("labels_", "")

        summary_rows.append({
            "model": model_tag,
            "n_samples": len(y_true),
            "alpha_vs_gold": round(alpha, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "mcc": round(mcc, 4),
        })

        for lbl in ALLOWED_LABELS:
            per_class_rows.append({
                "model": model_tag,
                "label": lbl,
                "precision": round(report.get(lbl, {}).get("precision", 0.0), 4),
                "recall": round(report.get(lbl, {}).get("recall", 0.0), 4),
                "f1": round(report.get(lbl, {}).get("f1-score", 0.0), 4),
                "support": report.get(lbl, {}).get("support", 0),
            })

        print(f"{model_tag} → Macro F1: {macro_f1:.3f} | MCC: {mcc:.3f} | Alpha: {alpha:.3f}")

    summary_df = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    per_class_df = pd.DataFrame(per_class_rows)

    summary_df.to_csv(OUT_DIR / "benchmark_results_summary.csv", index=False)
    per_class_df.to_csv(OUT_DIR / "benchmark_results_per_class.csv", index=False)

    print("\nBenchmark complete.")
    print(summary_df)


if __name__ == "__main__":
    main()