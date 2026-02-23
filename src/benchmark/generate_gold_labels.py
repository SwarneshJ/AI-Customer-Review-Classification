"""Generate a single gold label per row from multiple human annotators.

Default behavior: majority vote across provided label columns. Tie handling
can be set to 'first' (take the first annotator's label) or 'ambiguous'
(mark as 'AMBIGUOUS').

Writes output CSV to `data/processed/reviews_manual_1000_gold.csv` by default.
"""
import csv
import argparse
from collections import Counter
from typing import List, Optional


def read_rows(path: str, cols: List[str]):
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return rows


def majority_vote(labels: List[Optional[str]], tie_break='first') -> Optional[str]:
    vals = [l for l in labels if l is not None and l != '']
    if not vals:
        return None
    cnt = Counter(vals)
    most_common = cnt.most_common()
    if len(most_common) == 0:
        return None
    # check majority
    top_label, top_count = most_common[0]
    if top_count > 1:
        return top_label
    # no majority (all unique or tie)
    if tie_break == 'first':
        return vals[0]
    return 'AMBIGUOUS'


def write_rows(out_path: str, rows: List[dict], fieldnames: List[str]):
    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            # only include keys present in fieldnames
            out = {k: r.get(k, '') for k in fieldnames}
            writer.writerow(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='Input CSV path')
    p.add_argument('--cols', nargs='+', required=True, help='Annotator label columns')
    p.add_argument('--out', default='data/processed/reviews_manual_1000_gold.csv', help='Output CSV path')
    p.add_argument('--tie', choices=['first', 'ambiguous'], default='first', help='Tie-breaking strategy')
    args = p.parse_args()

    rows = read_rows(args.csv, args.cols)
    total = len(rows)
    tie_count = 0
    missing_count = 0
    counts = Counter()
    
    # Define columns to exclude from the final output
    columns_to_drop = {'score', 'userName', 'app', 'platform', ''}  # add any other metadata columns to drop as needed

    out_rows = []
    for r in rows:
        labels = [r.get(c, '').strip() or None for c in args.cols]
        gold = majority_vote(labels, tie_break=args.tie)
        if gold is None or gold == '':
            missing_count += 1
        if gold == 'AMBIGUOUS':
            tie_count += 1
        if gold not in (None, '', 'AMBIGUOUS'):
            counts[gold] += 1

        # produce a reduced row (drop annotator columns AND unwanted metadata) and add gold
        new_row = {k: v for k, v in r.items() if k not in args.cols and k not in columns_to_drop}
        new_row['gold_label'] = gold or ''
        out_rows.append(new_row)

    # determine output fieldnames: original fields without annotator cols & unwanted metadata + gold_label
    original_fieldnames = list(rows[0].keys()) if rows else []
    fieldnames = [f for f in original_fieldnames if f not in args.cols and f not in columns_to_drop] + ['gold_label']
    write_rows(args.out, out_rows, fieldnames)

    print(f'Wrote {len(out_rows)} rows to {args.out}')
    print(f'Total rows: {total}, missing gold: {missing_count}, ambiguous ties: {tie_count}')
    print('Top gold label counts:')
    for label, c in counts.most_common():
        print(f'  {label}: {c}')


if __name__ == '__main__':
    main()