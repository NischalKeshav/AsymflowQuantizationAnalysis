"""Side-by-side crop figure for one prompt: pixel bf16/int8/FP8/NF4 on top, latent bf16/NF4 below.

    python analysis/crop_figure.py --root out/pilot --csv results/full_all.csv --index 10 --out figures/crops_0010.jpg

Crop windows are chosen automatically: among 300x300 windows (stride 64) whose bf16 content has at
least median local detail, the one where NF4 changes the high-frequency energy the most. Pass
--pixel-box/--latent-box (x,y) to override.
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CELLS_PIX = [('bf16', 'pixel bf16'), ('int8wo', 'pixel int8'), ('float8dq', 'pixel FP8'), ('nf4', 'pixel NF4')]
CELLS_LAT = [('bf16', 'latent bf16'), ('nf4', 'latent NF4')]
TEAL, PINK, BG = (57, 197, 187), (242, 167, 199), (15, 26, 26)


def luma(img):
    a = np.asarray(img.convert('RGB'), dtype=np.float64) / 255.0
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def hf_energy(y, cutoff=0.5):
    spec = np.abs(np.fft.fftshift(np.fft.fft2(y - y.mean()))) ** 2
    h, w = y.shape
    fy = np.linspace(-1, 1, h)[:, None]
    fx = np.linspace(-1, 1, w)[None, :]
    return spec[np.sqrt(fy ** 2 + fx ** 2) > cutoff].sum()


def pick_window(ref, quant, size, stride=64, floor_q=50):
    yr, yq = luma(ref), luma(quant)
    h, w = yr.shape
    cands = []
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            er = hf_energy(yr[y:y + size, x:x + size])
            eq = hf_energy(yq[y:y + size, x:x + size])
            cands.append((x, y, er, eq / max(er, 1e-9)))
    floor = np.percentile([c[2] for c in cands], floor_q)
    good = [c for c in cands if c[2] >= floor] or cands
    x, y, _, _ = max(good, key=lambda c: abs(np.log(c[3])))
    return (x, y, x + size, y + size)


def font(bold, size):
    path = subprocess.check_output(['fc-match', '-f', '%{file}', 'DejaVu Sans' + (':bold' if bold else '')]).decode()
    return ImageFont.truetype(path, size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='dir containing <model>/<quant>/<idx>_seed<idx>.png')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--prompts', default='prompts/hpsv3_200.jsonl')
    ap.add_argument('--index', type=int, required=True)
    ap.add_argument('--size', type=int, default=300)
    ap.add_argument('--pixel-box', default=None, help='x,y of the pixel crop (default: automatic)')
    ap.add_argument('--latent-box', default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    i = args.index
    scores = {}
    for r in csv.DictReader(open(args.csv)):
        if int(r['image'][:4]) == i:
            scores[(r['model'], r['quant'])] = float(r['hf_ratio'])
    prompt = next(json.loads(l) for l in open(args.prompts) if json.loads(l)['index'] == i)

    def img(model, quant):
        return Image.open(Path(args.root) / model / quant / f'{i:04d}_seed{i}.png').convert('RGB')

    S = args.size
    if args.pixel_box:
        x, y = map(int, args.pixel_box.split(',')); pbox = (x, y, x + S, y + S)
    else:
        pbox = pick_window(img('asym', 'bf16'), img('asym', 'nf4'), S)
    if args.latent_box:
        x, y = map(int, args.latent_box.split(',')); lbox = (x, y, x + S, y + S)
    else:
        lbox = pick_window(img('latent', 'bf16'), img('latent', 'nf4'), S)

    U = 2; W = S * U; gap = 14; top = 76
    fb, fr, fsm = font(True, 30), font(False, 24), font(False, 20)
    canvas = Image.new('RGB', (4 * W + 5 * gap, 2 * (top + W) + 3 * gap + 40), BG)
    d = ImageDraw.Draw(canvas)

    def put(model, quant, box, col, row, label, color):
        c = img(model, quant).crop(box).resize((W, W), Image.LANCZOS)
        x = gap + col * (W + gap); y = gap + row * (top + W + gap)
        sub = 'reference' if quant == 'bf16' else f'HF {scores[(model, quant)]:.2f}x'
        d.text((x, y + 4), label, fill=color, font=fb); d.text((x, y + 42), sub, fill=(210, 210, 210), font=fr)
        canvas.paste(c, (x, y + top))

    for col, (q, label) in enumerate(CELLS_PIX):
        put('asym', q, pbox, col, 0, label, TEAL)
    for col, (q, label) in enumerate(CELLS_LAT):
        put('latent', q, lbox, col, 1, label, PINK)

    # thumbnails of the full bf16 images with the crop window marked
    tx = gap + 2 * (W + gap); ty = gap + (top + W + gap) + top
    for k, (model, box, color) in enumerate((('asym', pbox, TEAL), ('latent', lbox, PINK))):
        full = img(model, 'bf16')
        t = full.resize((W // 2 - 8, W // 2 - 8), Image.LANCZOS)
        dt = ImageDraw.Draw(t); s = t.width / full.width
        dt.rectangle([box[0] * s, box[1] * s, box[2] * s, box[3] * s], outline=color, width=3)
        canvas.paste(t, (tx + k * (W // 2 + 8), ty))
    d.text((tx, ty + W // 2 + 4), 'crop location in the bf16 images (pixel left, latent right)', fill=(200, 200, 200), font=fsm)
    d.multiline_text((tx, ty + W // 2 + 34),
                     'Same prompt, same seed, same text embedding.\nHF = high-frequency energy ratio vs the same\nmodel\'s bf16 output (whole image).',
                     fill=(225, 225, 225), font=fsm, spacing=5)
    txt = prompt['prompt'].replace('\n', ' ')
    txt = txt[:150] + ('…' if len(txt) > 150 else '')
    d.text((gap, canvas.height - 32), f'#{i} {prompt["category"]}: {txt}', fill=(170, 170, 170), font=fsm)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, quality=90)
    print(args.out, 'pixel box', pbox, 'latent box', lbox)


if __name__ == '__main__':
    main()
