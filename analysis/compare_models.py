"""Paired pixel-vs-latent comparison from a score CSV (same prompts, same seeds).

    python analysis/compare_models.py results/full_all.csv [--quant nf4]

For each quant setting present in both models, pairs images by prompt index and reports
median/mean per model, the mean paired difference (pixel - latent) with a bootstrap 95% CI,
and a sign test, for LPIPS and the HF-energy ratio.
"""
import argparse
import csv
from collections import defaultdict

import numpy as np


def boot_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def sign_test_p(d):
    from math import comb
    d = np.asarray(d); d = d[d != 0]
    n, k = len(d), int((d > 0).sum())
    k = max(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--quant', default=None)
    args = ap.parse_args()

    data = defaultdict(dict)  # (model, quant) -> index -> row
    for r in csv.DictReader(open(args.csv)):
        data[(r['model'], r['quant'])][int(r['image'][:4])] = r
    quants = sorted({q for (_, q) in data})
    for q in quants:
        if args.quant and q != args.quant:
            continue
        a, b = data.get(('asym', q), {}), data.get(('latent', q), {})
        common = sorted(set(a) & set(b))
        if not common:
            continue
        print(f'\n== {q}: pixel n={len(a)}, latent n={len(b)}, paired n={len(common)}')
        for metric in ('lpips', 'hf_ratio'):
            pa = np.array([float(a[i][metric]) for i in common])
            pl = np.array([float(b[i][metric]) for i in common])
            d = pa - pl
            lo, hi = boot_ci(d)
            print(f'{metric:9s} pixel median {np.median(pa):.3f} mean {pa.mean():.3f} | '
                  f'latent median {np.median(pl):.3f} mean {pl.mean():.3f} | '
                  f'paired diff {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  '
                  f'pixel>latent in {np.mean(d > 0):.0%}  sign-test p={sign_test_p(d):.2g}')


if __name__ == '__main__':
    main()
