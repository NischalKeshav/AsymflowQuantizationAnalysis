"""Per-category breakdown of fidelity scores (generalization check beyond the portrait-heavy pilot).

    python analysis/by_category.py results/full_partial.csv prompts/hpsv3_200.jsonl [--model asym] [--quant nf4]

Prints, for each (model, quant) cell: n, median/mean HF ratio, fraction of images with HF > 1.5,
median LPIPS, overall and per category, plus a bootstrap CI on the median HF ratio.
"""
import argparse
import csv
import json
from collections import defaultdict

import numpy as np


def boot_median_ci(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    meds = [np.median(rng.choice(x, len(x))) for _ in range(n)]
    return np.percentile(meds, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('prompts')
    ap.add_argument('--model', default=None)
    ap.add_argument('--quant', default=None)
    args = ap.parse_args()

    cat = {}
    for line in open(args.prompts):
        r = json.loads(line)
        cat[r['index']] = r['category']

    cells = defaultdict(list)
    for r in csv.DictReader(open(args.csv)):
        if args.model and r['model'] != args.model:
            continue
        if args.quant and r['quant'] != args.quant:
            continue
        idx = int(r['image'][:4])
        cells[(r['model'], r['quant'])].append(
            dict(index=idx, cat=cat[idx], lpips=float(r['lpips']), hf=float(r['hf_ratio']), psnr=float(r['psnr'])))

    for (model, quant), rows in sorted(cells.items()):
        hf = np.array([r['hf'] for r in rows]); lp = np.array([r['lpips'] for r in rows])
        lo, hi = boot_median_ci(hf)
        print(f'\n== {model}/{quant}: n={len(rows)}  HF median {np.median(hf):.2f} [{lo:.2f}, {hi:.2f}] '
              f'mean {hf.mean():.2f}  HF>1.5: {np.mean(hf > 1.5):.0%}  LPIPS median {np.median(lp):.3f}')
        by = defaultdict(list)
        for r in rows:
            by[r['cat']].append(r)
        print(f'{"category":16s} {"n":>3s} {"HF med":>7s} {"HF mean":>8s} {"HF>1.5":>7s} {"LPIPS":>6s}')
        for c, rs in sorted(by.items(), key=lambda kv: -np.median([r['hf'] for r in kv[1]])):
            h = np.array([r['hf'] for r in rs]); l = np.array([r['lpips'] for r in rs])
            print(f'{c:16s} {len(rs):3d} {np.median(h):7.2f} {h.mean():8.2f} {np.mean(h > 1.5):7.0%} {np.median(l):6.3f}')
        pilot = [r for r in rows if r['index'] < 20]; rest = [r for r in rows if r['index'] >= 20]
        if pilot and rest:
            print(f'pilot prompts (0-19): HF med {np.median([r["hf"] for r in pilot]):.2f} | '
                  f'new prompts (20+): HF med {np.median([r["hf"] for r in rest]):.2f} n={len(rest)}')


if __name__ == '__main__':
    main()
