"""Evaluate model predictions against gold labels.

Computes:
- Krippendorff's alpha (nominal) between gold and predictions (or multiple prediction columns)
- Precision / Recall / F1 (per-class and macro)
- Matthews correlation coefficient (multiclass)
- AUC (binary or one-vs-rest if probability columns provided)

Usage examples:
  python src/evaluate_predictions.py \
    --gold data/processed/reviews_manual_1000_gold.csv \
    --pred data/processed/reviews_manual_1000_gold.csv \
    --gold-col gold_label --pred-col predicted_label

  python src/evaluate_predictions.py \
    --gold data/processed/reviews_manual_1000_gold.csv \
    --pred model_outputs.csv \
    --gold-col gold_label --pred-col modelA_pred --prob-prefix modelA_prob_

"""
from collections import Counter, defaultdict
import argparse
import csv
import math
from typing import List, Optional, Dict

import pandas as pd

from benchmark.krippendorff_alpha import krippendorff_alpha_nominal


def align_data(gold_df: pd.DataFrame, pred_df: pd.DataFrame, id_col: Optional[str]):
    if id_col:
        merged = gold_df.merge(pred_df, on=id_col, suffixes=('_gold', '_pred'))
    else:
        # align by index
        merged = gold_df.copy()
        merged = merged.reset_index(drop=True)
        pred_df2 = pred_df.reset_index(drop=True)
        merged = pd.concat([merged, pred_df2], axis=1)
    return merged


def confusion_matrix(true: List[str], pred: List[str], labels: List[str]) -> List[List[int]]:
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    C = [[0] * k for _ in range(k)]
    for t, p in zip(true, pred):
        if t not in idx or p not in idx:
            continue
        C[idx[t]][idx[p]] += 1
    return C


def precision_recall_f1_from_confusion(C: List[List[int]], labels: List[str]):
    k = len(labels)
    precisions = {}
    recalls = {}
    f1s = {}
    supports = {}
    for i, lab in enumerate(labels):
        tp = C[i][i]
        fn = sum(C[i][j] for j in range(k) if j != i)
        fp = sum(C[j][i] for j in range(k) if j != i)
        support = tp + fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions[lab] = prec
        recalls[lab] = rec
        f1s[lab] = f1
        supports[lab] = support
    # macro
    macro_f1 = sum(f1s.values()) / k if k > 0 else 0.0
    return precisions, recalls, f1s, supports, macro_f1


def matthews_corrcoef_multiclass(C: List[List[int]]):
    # multiclass MCC formula
    k = len(C)
    t_k = [sum(C[i][k_] for i in range(k)) for k_ in range(k)]  # true counts per class (col sums)
    p_k = [sum(C[k_][j] for j in range(k)) for k_ in range(k)]  # pred counts per class (row sums)
    c = sum(C[i][i] for i in range(k))
    s = sum(sum(row) for row in C)
    sum_pk_tk = sum(p_k[i] * t_k[i] for i in range(k))
    num = c * s - sum_pk_tk
    denom_term1 = s * s - sum(p * p for p in p_k)
    denom_term2 = s * s - sum(t * t for t in t_k)
    denom = math.sqrt(denom_term1 * denom_term2)
    if denom == 0:
        return 0.0
    return num / denom


def compute_auc_binary(scores: List[float], true: List[int]):
    # ranks-based AUC (equivalent to Mann-Whitney U)
    paired = sorted(zip(scores, true), key=lambda x: (x[0], x[1]))
    n_pos = sum(true)
    n_neg = len(true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    # assign ranks with average for ties
    ranks = []
    i = 0
    N = len(paired)
    while i < N:
        j = i
        while j + 1 < N and paired[j + 1][0] == paired[i][0]:
            j += 1
        # average rank for positions i..j
        avg_rank = (i + 1 + j + 1) / 2.0
        for _ in range(i, j + 1):
            ranks.append(avg_rank)
        i = j + 1
    # sum ranks for positive
    sum_ranks_pos = 0.0
    for (r, t), rank in zip(paired, ranks):
        if t == 1:
            sum_ranks_pos += rank
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


def evaluate_one(true_labels: List[str], pred_labels: List[str], prob_scores: Optional[List[float]] = None):
    # coerce labels to strings to avoid mixing types (NaN floats etc.)
    true_s = [str(x) if x is not None and not (isinstance(x, float) and math.isnan(x)) else '' for x in true_labels]
    pred_s = [str(x) if x is not None and not (isinstance(x, float) and math.isnan(x)) else '' for x in pred_labels]
    labels = sorted(list(set(true_s) | set(pred_s)))
    C = confusion_matrix(true_labels, pred_labels, labels)
    precisions, recalls, f1s, supports, macro_f1 = precision_recall_f1_from_confusion(C, labels)
    mcc = matthews_corrcoef_multiclass(C)
    auc = None
    if prob_scores is not None and len(labels) == 2:
        # map true labels to {0,1} where positive is labels[1]
        pos_label = labels[1]
        true_bin = [1 if (str(t) == pos_label) else 0 for t in true_labels]
        auc = compute_auc_binary(prob_scores, true_bin)
    return {
        'labels': labels,
        'confusion': C,
        'precision': precisions,
        'recall': recalls,
        'f1': f1s,
        'support': supports,
        'f1_macro': macro_f1,
        'mcc': mcc,
        'auc': auc,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gold', required=True, help='Gold CSV path')
    p.add_argument('--pred', required=True, help='Predictions CSV path')
    p.add_argument('--gold-col', default='gold_label', help='Column name for gold labels')
    p.add_argument('--pred-col', default='predicted_label', help='Column name for predicted labels')
    p.add_argument('--id-col', default=None, help='Optional id column to align on')
    p.add_argument('--prob-col', default=None, help='Optional probability column for positive class (binary)')
    args = p.parse_args()

    gold_df = pd.read_csv(args.gold)
    pred_df = pd.read_csv(args.pred)
    merged = align_data(gold_df, pred_df, args.id_col)

    if args.gold_col not in merged.columns:
        raise SystemExit(f'gold column {args.gold_col} not found in merged data')
    if args.pred_col not in merged.columns:
        raise SystemExit(f'pred column {args.pred_col} not found in merged data')

    true = merged[args.gold_col].astype(str).tolist()
    pred = merged[args.pred_col].astype(str).tolist()
    prob_scores = None
    if args.prob_col and args.prob_col in merged.columns:
        prob_scores = merged[args.prob_col].astype(float).tolist()

    # Krippendorff's alpha between gold and pred per unit
    paired = [[t, p] for t, p in zip(true, pred)]
    alpha = krippendorff_alpha_nominal(paired)

    metrics = evaluate_one(true, pred, prob_scores)

    print('Krippendorff\'s alpha (gold vs pred):', alpha)
    print('Macro F1:', metrics['f1_macro'])
    print('MCC:', metrics['mcc'])
    if metrics['auc'] is not None:
        print('AUC:', metrics['auc'])
    print('\nPer-class F1:')
    for lab, f in metrics['f1'].items():
        print(f'  {lab}: f1={f:.4f} (support={metrics["support"][lab]})')


if __name__ == '__main__':
    main()
