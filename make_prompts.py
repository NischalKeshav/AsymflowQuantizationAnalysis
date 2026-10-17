"""Sample a fixed, category-stratified prompt set from the HPSv3 (HPDv3) benchmark.

Uses the same 12 benchmark files as LakonLab's AsymFLUX.2 klein eval config.

    python make_prompts.py --n 200 --out prompts/hpsv3_200.jsonl
"""
import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

CATEGORIES = ['Characters', 'Arts', 'Design', 'Architecture', 'Animals', 'Natural Scenery',
              'Transportation', 'Products', 'Others', 'Plants', 'Food', 'Science']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    per_cat = [args.n // len(CATEGORIES) + (i < args.n % len(CATEGORIES))
               for i in range(len(CATEGORIES))]
    rows = []
    for cat, k in zip(CATEGORIES, per_cat):
        path = hf_hub_download('MizzenAI/HPDv3', f'benchmark/benchmark_{cat}.json',
                               repo_type='dataset')
        captions = sorted({item['caption'].strip() for item in json.load(open(path))
                           if item.get('caption', '').strip()})
        for caption in rng.sample(captions, k):
            rows.append({'category': cat, 'prompt': caption})
        print(f'{cat}: {k} of {len(captions)}')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        for i, row in enumerate(rows):
            f.write(json.dumps({'index': i, **row}) + '\n')
    print(f'wrote {len(rows)} prompts to {args.out}')


if __name__ == '__main__':
    main()
