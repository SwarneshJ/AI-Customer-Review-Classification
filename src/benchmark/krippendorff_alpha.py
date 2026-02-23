"""Compute Krippendorff's alpha (nominal) for a CSV of human labels.

Usage:
    python src/krippendorff_alpha.py \
        --csv data/raw/reviews_manual_1000_labeled.csv \
        --cols "Label 1 (Kenneth)" "Label 2 (Ben)" "Label 3 (Sahil)"

The script is dependency-free and implements the nominal disagreement.
"""
from collections import Counter, defaultdict
import csv
import math
import argparse
from typing import List, Iterable, Tuple, Dict


def read_ratings(csv_path: str, columns: List[str]) -> List[List[str]]:
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            vals = [r.get(c, '').strip() or None for c in columns]
            rows.append(vals)
    return rows


def krippendorff_alpha_nominal(data: Iterable[Iterable[str]]) -> float:
    """Compute Krippendorff's alpha for nominal data.

    data: iterable of units, each unit is an iterable of category labels
          (use None or empty for missing).

    Implementation follows the coincidence-matrix approach.
    """
    # Build coincidence counts over ordered pairs r != s
    O: Dict[Tuple[str, str], float] = Counter()
    categories = set()
    total_pairs = 0.0

    for unit in data:
        # collect non-missing labels for this unit
        labels = [v for v in unit if v is not None and v != '']
        m = len(labels)
        if m < 2:
            continue
        for i in range(m):
            a = labels[i]
            categories.add(a)
            for j in range(m):
                if i == j:
                    continue
                b = labels[j]
                categories.add(b)
                O[(a, b)] += 1.0
                total_pairs += 1.0

    if total_pairs == 0:
        return float('nan')

    # Marginals
    o_dot = defaultdict(float)
    for (a, b), cnt in O.items():
        o_dot[a] += cnt

    # Observed disagreement Do (proportion)
    Do = 0.0
    for (a, b), cnt in O.items():
        d = 0.0 if a == b else 1.0
        Do += cnt * d
    Do = Do / total_pairs

    # Expected disagreement De
    # Expected coincidence E_ab = o_dot[a] * o_dot[b] / total_pairs
    # De = sum_{a,b} E_ab * delta(a,b) / total_pairs
    De_num = 0.0
    for a in o_dot.keys():
        for b in o_dot.keys():
            if a == b:
                continue
            De_num += (o_dot[a] * o_dot[b])
    De = De_num / (total_pairs * total_pairs)

    if De == 0.0:
        # Perfect agreement or no variability
        return 1.0

    alpha = 1.0 - (Do / De)
    return alpha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='Path to input CSV')
    p.add_argument('--cols', nargs='+', required=True, help='Label columns (in order)')
    args = p.parse_args()

    rows = read_ratings(args.csv, args.cols)
    alpha = krippendorff_alpha_nominal(rows)
    print(f"Krippendorff's alpha (nominal) for columns {args.cols}: {alpha:.6f}")


if __name__ == '__main__':
    main()
