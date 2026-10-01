# -*- coding: utf-8 -*-
"""Per-class training distribution + per-class error distribution.

Two independent modes; both write de-identifiable summary CSVs that can go
straight into a paper (no images, no raw predictions).

Mode A -- training distribution (answers "median / long tail"):
    input: either
      (1) a root directory whose immediate sub-directories are class ids, or
      (2) a label file (csv/txt) with columns  path,label  (header optional)

Mode B -- error distribution (answers "per-class accuracy / confusion pairs"):
    input: a prediction file (csv) with columns  gt,pred  (one row per crop).
           Optional extra column  count  for pre-aggregated pairs.

Usage
-----
python per_class_stats.py train --root /path/to/train_crops
python per_class_stats.py train --labels /path/train_list.csv
python per_class_stats.py errors --preds /path/pred_vs_gt.csv
python per_class_stats.py errors --preds p.csv --train-counts counts.csv

Outputs are written next to the input as *_summary.csv / *_summary.txt.
Stdlib only.
"""
import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict


def pct(x, n):
    return 0.0 if n == 0 else 100.0 * x / n


def quantile(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


# ----------------------------------------------------------------- Mode A
def load_counts_from_root(root):
    counts = Counter()
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if os.path.isdir(sub):
            n = sum(1 for f in os.listdir(sub)
                    if os.path.isfile(os.path.join(sub, f)))
            if n:
                counts[name] = n
    return counts


def load_counts_from_jsonl(path, key='char', where=None):
    """Read a JSON-Lines metadata file (one record per image).

    key    -- the field that identifies the category ('char' or 'label')
    where  -- optional "field=value" filter applied to every record
    """
    counts = Counter()
    filt = None
    if where:
        f, v = where.split('=', 1)
        filt = (f.strip(), v.strip())
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if filt and str(rec.get(filt[0])) != filt[1]:
                continue
            k = rec.get(key)
            if k is None:
                continue
            counts[str(k)] += 1
    return counts


def load_counts_from_labels(path):
    counts = Counter()
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.replace('\t', ',').split(',')]
            if len(parts) < 2:
                continue
            lab = parts[-1]
            add = 1
            # tolerate "path,label,count": if the last field is a bare number,
            # the label sits one field left and the number is the count
            if len(parts) >= 3 and lab.isdigit():
                lab = parts[-2]
                add = int(parts[-1])
            if lab.lower() in ('label', 'class', 'category', 'id'):
                continue          # header
            counts[lab] += add
    return counts


def report_train(counts, out_prefix):
    vals = sorted(counts.values())
    n_cls = len(vals)
    total = sum(vals)
    mean = total / n_cls if n_cls else 0.0

    rows = [
        ('categories', n_cls),
        ('training images', total),
        ('mean per category', round(mean, 2)),
        ('median per category', round(statistics.median(vals), 1) if vals else 0),
        ('min', vals[0] if vals else 0),
        ('max', vals[-1] if vals else 0),
        ('p10', round(quantile(vals, 0.10), 1)),
        ('p25', round(quantile(vals, 0.25), 1)),
        ('p75', round(quantile(vals, 0.75), 1)),
        ('p90', round(quantile(vals, 0.90), 1)),
        ('categories with <5 images', '%d (%.1f%%)' % (
            sum(1 for v in vals if v < 5), pct(sum(1 for v in vals if v < 5), n_cls))),
        ('categories with <10 images', '%d (%.1f%%)' % (
            sum(1 for v in vals if v < 10), pct(sum(1 for v in vals if v < 10), n_cls))),
        ('categories with <20 images', '%d (%.1f%%)' % (
            sum(1 for v in vals if v < 20), pct(sum(1 for v in vals if v < 20), n_cls))),
        ('share of images held by top 10% categories', '%.1f%%' % pct(
            sum(vals[-max(1, n_cls // 10):]), total)),
        ('share of images held by top 1% categories', '%.1f%%' % pct(
            sum(vals[-max(1, n_cls // 100):]), total)),
    ]

    txt = out_prefix + '_train_summary.txt'
    with open(txt, 'w', encoding='utf-8') as f:
        f.write('Per-category training distribution\n')
        for k, v in rows:
            f.write('%-45s %s\n' % (k, v))

    # de-identified histogram: safe to publish as supplementary
    csv_path = out_prefix + '_train_histogram.csv'
    bins = [(1, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, 199), (200, 499),
            (500, 999), (1000, 10 ** 9)]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['images_per_category_bin', 'categories', 'share_of_categories_pct'])
        for lo, hi in bins:
            c = sum(1 for v in vals if lo <= v <= hi)
            w.writerow(['%d-%s' % (lo, 'inf' if hi > 10 ** 8 else hi), c,
                        '%.2f' % pct(c, n_cls)])

    print(open(txt, encoding='utf-8').read())
    print('written:', txt, '|', csv_path)


# ----------------------------------------------------------------- Mode B
def load_pairs(path):
    pairs = []
    with open(path, encoding='utf-8', errors='replace') as f:
        first = True
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.replace('\t', ',').split(',')]
            if first and parts and parts[0].lower() in ('gt', 'true', 'label', 'gold'):
                first = False
                continue
            first = False
            if len(parts) < 2:
                continue
            n = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
            pairs.append((parts[0], parts[1], n))
    return pairs


def report_errors(pairs, train_counts, out_prefix):
    per_cls_total = Counter()
    per_cls_correct = Counter()
    conf = Counter()
    for gt, pred, n in pairs:
        per_cls_total[gt] += n
        if gt == pred:
            per_cls_correct[gt] += n
        else:
            conf[(gt, pred)] += n

    accs = [per_cls_correct[c] / per_cls_total[c] for c in per_cls_total]
    accs_sorted = sorted(accs)
    n_cls = len(accs)

    rows = [
        ('categories observed', n_cls),
        ('crops evaluated', sum(per_cls_total.values())),
        ('macro accuracy (unweighted mean over categories)',
         '%.4f' % (sum(accs) / n_cls if n_cls else 0)),
        ('micro accuracy (over crops)',
         '%.4f' % (sum(per_cls_correct.values()) / sum(per_cls_total.values()))),
        ('median per-category accuracy',
         '%.4f' % (statistics.median(accs) if accs else 0)),
        ('categories with accuracy = 0', '%d (%.1f%%)' % (
            sum(1 for a in accs if a == 0), pct(sum(1 for a in accs if a == 0), n_cls))),
        ('categories with accuracy >= 0.9', '%d (%.1f%%)' % (
            sum(1 for a in accs if a >= 0.9), pct(sum(1 for a in accs if a >= 0.9), n_cls))),
        ('accuracy p25 / p75', '%.4f / %.4f' % (
            quantile(accs_sorted, 0.25), quantile(accs_sorted, 0.75))),
    ]

    # head / medium / tail split by training frequency, if available
    bucket_rows = []
    if train_counts:
        def bucket(c):
            n = train_counts.get(c, 0)
            if n < 10:
                return 'tail (<10 training images)'
            if n < 50:
                return 'medium (10-49)'
            return 'head (>=50)'
        agg = defaultdict(lambda: [0, 0, 0])   # crops, correct, categories
        for c in per_cls_total:
            b = bucket(c)
            agg[b][0] += per_cls_total[c]
            agg[b][1] += per_cls_correct[c]
            agg[b][2] += 1
        for b in ('head (>=50)', 'medium (10-49)', 'tail (<10 training images)'):
            if b in agg:
                crops, corr, cls = agg[b]
                bucket_rows.append((b, cls, crops, '%.4f' % (corr / crops)))

    txt = out_prefix + '_error_summary.txt'
    with open(txt, 'w', encoding='utf-8') as f:
        f.write('Per-category error distribution\n')
        for k, v in rows:
            f.write('%-48s %s\n' % (k, v))
        if bucket_rows:
            f.write('\nAccuracy by training-frequency band\n')
            f.write('%-30s %10s %10s %10s\n' % ('band', 'categories', 'crops', 'accuracy'))
            for b, cls, crops, a in bucket_rows:
                f.write('%-30s %10d %10d %10s\n' % (b, cls, crops, a))

    csv_path = out_prefix + '_error_by_band.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['band', 'categories', 'crops', 'accuracy'])
        for b, cls, crops, a in bucket_rows:
            w.writerow([b, cls, crops, a])

    # top confusion pairs -- de-identified: only the pair and the count
    conf_csv = out_prefix + '_top_confusions.csv'
    with open(conf_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['reference_category', 'predicted_category', 'count'])
        for (gt, pred), n in conf.most_common(50):
            w.writerow([gt, pred, n])

    print(open(txt, encoding='utf-8').read())
    print('written:', txt, '|', csv_path, '|', conf_csv)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='mode', required=True)

    a = sub.add_parser('train')
    a.add_argument('--root')
    a.add_argument('--labels')
    a.add_argument('--jsonl')
    a.add_argument('--key', default='char')
    a.add_argument('--where')
    a.add_argument('--out', default='per_class')

    b = sub.add_parser('errors')
    b.add_argument('--preds', required=True)
    b.add_argument('--train-counts')
    b.add_argument('--out', default='per_class')

    args = ap.parse_args()

    if args.mode == 'train':
        if args.root:
            counts = load_counts_from_root(args.root)
        elif args.labels:
            counts = load_counts_from_labels(args.labels)
        elif args.jsonl:
            counts = load_counts_from_jsonl(args.jsonl, args.key, args.where)
        else:
            sys.exit('need --root or --labels')
        if not counts:
            sys.exit('no counts loaded')
        report_train(counts, args.out)
    else:
        pairs = load_pairs(args.preds)
        tc = {}
        if args.train_counts:
            tc = dict(load_counts_from_labels(args.train_counts))
        if not pairs:
            sys.exit('no pairs loaded')
        report_errors(pairs, tc, args.out)


if __name__ == '__main__':
    main()
