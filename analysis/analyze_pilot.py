"""Sanity analysis of the pilot: is the pixel-vs-latent texture effect real, metric-robust,
outlier-free, and detectable at n=200?"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats

ROOT = Path(__file__).parent / 'data'
PILOT = ROOT / 'pilot'
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- image features
def luma(path):
    img = np.asarray(Image.open(path).convert('RGB'), dtype=np.float64) / 255.0
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def features(path):
    y = luma(path)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(y - y.mean()))) ** 2
    h, w = y.shape
    fy = np.linspace(-1, 1, h)[:, None]
    fx = np.linspace(-1, 1, w)[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    total = spec.sum()
    bands = {
        'lo<0.25': spec[r < 0.25].sum(),
        'mid0.25-0.5': spec[(r >= 0.25) & (r < 0.5)].sum(),
        'hi0.5-0.75': spec[(r >= 0.5) & (r < 0.75)].sum(),
        'top>0.75': spec[r >= 0.75].sum(),
    }
    hf = bands['hi0.5-0.75'] + bands['top>0.75']  # same as score_fidelity cutoff 0.5
    # robust noise sigma from finest-scale diagonal wavelet detail (Donoho)
    d = (y[0::2, 0::2] - y[1::2, 0::2] - y[0::2, 1::2] + y[1::2, 1::2]) / 2
    sigma = np.median(np.abs(d - np.median(d))) / 0.6745
    lap = (-4 * y[1:-1, 1:-1] + y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:])
    return dict(total=total, hf=hf, hf_frac=hf / total, sigma=sigma, lapvar=lap.var(), **bands)


cache = {}
def feat(model, quant, name):
    key = (model, quant, name)
    if key not in cache:
        cache[key] = features(PILOT / model / quant / name)
    return cache[key]


# ---------------------------------------------------------------- load scores + prompts
rows = list(csv.DictReader(open(ROOT / 'pilot_fidelity.csv')))
for r in rows:
    r['lpips'] = float(r['lpips']); r['psnr'] = float(r['psnr']); r['hf_ratio'] = float(r['hf_ratio'])
    r['idx'] = int(r['image'][:4])
prompts = {json.loads(l)['index']: json.loads(l) for l in open(ROOT / 'hpsv3_200.jsonl')}
cells = defaultdict(list)
for r in rows:
    cells[(r['model'], r['quant'])].append(r)
for v in cells.values():
    v.sort(key=lambda r: r['idx'])


def boot_ci(x, f=np.mean, n=5000):
    x = np.asarray(x)
    s = np.array([f(rng.choice(x, len(x))) for _ in range(n)])
    return np.percentile(s, [2.5, 97.5])


def trimmed(x, k=2):
    x = np.sort(x)
    return x[k:-k].mean()


print('=' * 78)
print('1. PILOT PROMPT COMPOSITION (indices 0-19)')
cats = defaultdict(int)
for i in range(20):
    cats[prompts[i]['category']] += 1
print('  ', dict(cats))
print('   -> the prompt file is sorted by category; the pilot is NOT stratified.')

print('=' * 78)
print('2. PER-CELL ROBUSTNESS: mean / median / trimmed(drop 2 each end) / frac HF>1.5 / frac LPIPS>0.4')
for (m, q), v in sorted(cells.items()):
    hf = np.array([r['hf_ratio'] for r in v]); lp = np.array([r['lpips'] for r in v])
    print(f'   {m:6s} {q:9s} HF mean {hf.mean():5.2f} med {np.median(hf):5.2f} trim {trimmed(hf):5.2f} '
          f'min {hf.min():4.2f} max {hf.max():5.2f} | >1.5: {np.mean(hf > 1.5):.0%} | '
          f'LPIPS mean {lp.mean():.3f} med {np.median(lp):.3f} >0.4: {np.mean(lp > 0.4):.0%}')

print('=' * 78)
print('3. PAIRED PIXEL - LATENT (same prompt/seed), Wilcoxon + sign count + bootstrap CI')
for q in ('nf4', 'int8wo'):
    a = {r['idx']: r for r in cells[('asym', q)]}; b = {r['idx']: r for r in cells[('latent', q)]}
    common = sorted(set(a) & set(b))
    for metric in ('hf_ratio', 'lpips'):
        d = np.array([a[i][metric] - b[i][metric] for i in common])
        w = stats.wilcoxon(d)
        lo, hi = boot_ci(d)
        print(f'   {q:7s} {metric:8s} n={len(d)} mean diff {d.mean():+.3f} CI [{lo:+.3f},{hi:+.3f}] '
              f'median {np.median(d):+.3f} pixel>latent in {np.sum(d > 0)}/{len(d)}  Wilcoxon p={w.pvalue:.4f}')
    # also log-ratio of HF (ratio metrics should be compared in log space)
    d = np.array([np.log(a[i]['hf_ratio']) - np.log(b[i]['hf_ratio']) for i in common])
    lo, hi = boot_ci(d)
    print(f'   {q:7s} log-HF   mean diff {d.mean():+.3f} CI [{lo:+.3f},{hi:+.3f}] -> geometric ratio x{np.exp(d.mean()):.2f}')

print('=' * 78)
print('4. METRIC VALIDITY: is HF ratio driven by a small denominator (smooth bf16 images)?')
for m in ('asym', 'latent'):
    v = cells[(m, 'nf4')]
    base = np.array([feat(m, 'bf16', r['image'])['hf_frac'] for r in v])
    hf = np.array([r['hf_ratio'] for r in v])
    rho = stats.spearmanr(base, hf)
    print(f'   {m:6s} nf4: Spearman(bf16 HF fraction, HF ratio) = {rho.statistic:+.2f} (p={rho.pvalue:.3f})')
base_a = np.array([feat('asym', 'bf16', r['image'])['hf_frac'] for r in cells[('asym', 'nf4')]])
base_l = np.array([feat('latent', 'bf16', r['image'])['hf_frac'] for r in cells[('latent', 'nf4')]])
print(f'   baseline HF fraction of total energy: pixel bf16 {base_a.mean():.4f}  latent bf16 {base_l.mean():.4f} '
      f'(ratio {base_a.mean() / base_l.mean():.2f}; paired Wilcoxon p={stats.wilcoxon(base_a - base_l).pvalue:.3f})')

print('=' * 78)
print('5. WHERE DOES THE ADDED ENERGY LIVE? per-band quant/bf16 ratio (geometric mean over images)')
print('   ' + ' ' * 17 + ''.join(f'{b:>14s}' for b in ('lo<0.25', 'mid0.25-0.5', 'hi0.5-0.75', 'top>0.75', 'sigma', 'lapvar')))
for (m, q), v in sorted(cells.items()):
    out = []
    for key in ('lo<0.25', 'mid0.25-0.5', 'hi0.5-0.75', 'top>0.75', 'sigma', 'lapvar'):
        lr = [np.log(feat(m, q, r['image'])[key] / feat(m, 'bf16', r['image'])[key]) for r in v]
        out.append(np.exp(np.mean(lr)))
    print(f'   {m:6s} {q:9s}' + ''.join(f'{x:14.2f}' for x in out))
print('   sigma = robust pixel-noise estimate (finest-scale wavelet MAD); lapvar = Laplacian variance.')

print('=' * 78)
print('6. IS TEXTURE INDEPENDENT OF SEMANTIC DRIFT? Spearman(LPIPS, HF ratio) within cell')
for (m, q), v in sorted(cells.items()):
    rho = stats.spearmanr([r['lpips'] for r in v], [r['hf_ratio'] for r in v])
    print(f'   {m:6s} {q:9s} rho={rho.statistic:+.2f} p={rho.pvalue:.3f}')

print('=' * 78)
print('7. CATEGORY SPLIT (only Characters n=17 vs Arts n=3 available in the pilot)')
for (m, q) in (('asym', 'nf4'), ('latent', 'nf4'), ('asym', 'float8dq')):
    by = defaultdict(list)
    for r in cells[(m, q)]:
        by[prompts[r['idx']]['category']].append(r['hf_ratio'])
    print(f'   {m:6s} {q:9s} ' + '  '.join(f'{c}: HF med {np.median(x):.2f} (n={len(x)})' for c, x in by.items()))

print('=' * 78)
print('8. POWER AT n=200 (paired pixel-latent, two-sided alpha .05, 80% power), from pilot SD')
for q in ('nf4', 'int8wo'):
    a = {r['idx']: r for r in cells[('asym', q)]}; b = {r['idx']: r for r in cells[('latent', q)]}
    common = sorted(set(a) & set(b))
    for metric in ('hf_ratio', 'lpips'):
        d = np.array([a[i][metric] - b[i][metric] for i in common])
        sd = d.std(ddof=1)
        mde = 2.8 * sd / np.sqrt(200)
        power = stats.norm.cdf(abs(d.mean()) / (sd / np.sqrt(200)) - 1.96)
        print(f'   {q:7s} {metric:8s} pilot diff {d.mean():+.3f} sd {sd:.3f} -> MDE at n=200: {mde:.3f}; '
              f'power for pilot effect: {power:.2f}')

print('=' * 78)
print('9. SEMANTIC-DRIFT SANITY: PSNR vs LPIPS agreement, and worst images per cell')
for (m, q), v in sorted(cells.items()):
    worst = sorted(v, key=lambda r: -r['lpips'])[:3]
    print(f'   {m:6s} {q:9s} worst LPIPS: ' + ', '.join(f"{r['idx']:02d}({r['lpips']:.2f},HF{r['hf_ratio']:.1f})" for r in worst))

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(12, 9))
# (a) paired HF ratio pixel vs latent under nf4
a = {r['idx']: r for r in cells[('asym', 'nf4')]}; b = {r['idx']: r for r in cells[('latent', 'nf4')]}
common = sorted(set(a) & set(b))
for i in common:
    ax[0, 0].plot([0, 1], [b[i]['hf_ratio'], a[i]['hf_ratio']], 'o-', color='gray', alpha=.6)
ax[0, 0].set_xticks([0, 1]); ax[0, 0].set_xticklabels(['latent NF4', 'pixel NF4']); ax[0, 0].set_yscale('log')
ax[0, 0].axhline(1, ls='--', c='k'); ax[0, 0].set_ylabel('HF energy ratio vs bf16 (log)'); ax[0, 0].set_title('(a) same prompt+seed, NF4')
# (b) LPIPS vs HF ratio
for (m, q), v in sorted(cells.items()):
    ax[0, 1].scatter([r['lpips'] for r in v], [r['hf_ratio'] for r in v], label=f'{m}/{q}', alpha=.7,
                     marker='o' if m == 'asym' else 'x')
ax[0, 1].set_yscale('log'); ax[0, 1].axhline(1, ls='--', c='k'); ax[0, 1].set_xlabel('LPIPS vs bf16'); ax[0, 1].set_ylabel('HF ratio (log)')
ax[0, 1].legend(fontsize=7); ax[0, 1].set_title('(b) texture vs semantic drift')
# (c) per-band ratios
bands = ('lo<0.25', 'mid0.25-0.5', 'hi0.5-0.75', 'top>0.75')
for (m, q), v in sorted(cells.items()):
    ys = [np.exp(np.mean([np.log(feat(m, q, r['image'])[k] / feat(m, 'bf16', r['image'])[k]) for r in v])) for k in bands]
    ax[1, 0].plot(range(4), ys, 'o-' if m == 'asym' else 'x--', label=f'{m}/{q}')
ax[1, 0].set_xticks(range(4)); ax[1, 0].set_xticklabels(bands); ax[1, 0].set_yscale('log'); ax[1, 0].axhline(1, ls='--', c='k')
ax[1, 0].set_ylabel('quant/bf16 band energy (geo-mean)'); ax[1, 0].legend(fontsize=7); ax[1, 0].set_title('(c) spectral band of the added energy')
# (d) LPIPS distributions
labels, data = [], []
for (m, q), v in sorted(cells.items()):
    labels.append(f'{m}\n{q}'); data.append([r['lpips'] for r in v])
ax[1, 1].boxplot(data, tick_labels=labels); ax[1, 1].set_ylabel('LPIPS vs bf16'); ax[1, 1].set_title('(d) LPIPS per cell')
ax[1, 1].tick_params(axis='x', labelsize=7)
plt.tight_layout()
out = Path(__file__).parent / 'pilot_sanity.png'
plt.savefig(out, dpi=110)
print('figure ->', out)
