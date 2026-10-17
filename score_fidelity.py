"""Score quantized images against their same-seed bf16 twins.

For every image in out/<run>/<model>/<quant>/ that has a twin in .../bf16/, writes one CSV row:
LPIPS (AlexNet), PSNR, and the high-frequency energy ratio (quantized / bf16, FFT energy
above a radius cutoff). Ratio < 1 means fine texture was lost.

    python score_fidelity.py --run out/pilot --out results/pilot_fidelity.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0)


def psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)


def high_freq_energy(img, cutoff):
    """Fraction-free HF energy: sum of |FFT|^2 of luminance outside `cutoff` x Nyquist."""
    y = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).double()
    spec = torch.fft.fftshift(torch.fft.fft2(y - y.mean())).abs() ** 2
    h, w = y.shape
    fy = torch.linspace(-1, 1, h).view(-1, 1)
    fx = torch.linspace(-1, 1, w).view(1, -1)
    return spec[(fy ** 2 + fx ** 2).sqrt() > cutoff].sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='e.g. out/pilot (contains <model>/<quant>/)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cutoff', type=float, default=0.5)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--ref', default='bf16', help='reference subdir name under each <model>/')
    args = ap.parse_args()

    import lpips
    if args.device == 'cuda':
        from common import limit_vram
        limit_vram(4.0, min_gb=1.0)
    net = lpips.LPIPS(net='alex', verbose=False).to(args.device).eval()

    rows = []
    for model_dir in sorted(p for p in Path(args.run).iterdir() if p.is_dir()):
        ref_dir = model_dir / args.ref
        if not ref_dir.is_dir():
            continue
        for quant_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and p.name != args.ref):
            for img_path in sorted(quant_dir.glob('*.png')):
                ref_path = ref_dir / img_path.name
                if not ref_path.exists():
                    continue
                a, b = load(img_path), load(ref_path)
                with torch.no_grad():
                    to_net = lambda x: (x.permute(2, 0, 1)[None] * 2 - 1).to(args.device)
                    lp = net(to_net(a), to_net(b)).item()
                rows.append(dict(
                    model=model_dir.name, quant=quant_dir.name, image=img_path.name,
                    lpips=round(lp, 5), psnr=round(psnr(a, b), 3),
                    hf_ratio=round(high_freq_energy(a, args.cutoff) / high_freq_energy(b, args.cutoff), 5)))
            done = [r for r in rows if r['model'] == model_dir.name and r['quant'] == quant_dir.name]
            if done:
                print(f"{model_dir.name}/{quant_dir.name}: n={len(done)} "
                      f"LPIPS={np.mean([r['lpips'] for r in done]):.4f} "
                      f"PSNR={np.mean([r['psnr'] for r in done]):.2f} "
                      f"HF={np.mean([r['hf_ratio'] for r in done]):.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {len(rows)} rows to {args.out}')


if __name__ == '__main__':
    main()
