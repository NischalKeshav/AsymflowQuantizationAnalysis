"""Encode prompts once with the Qwen3 text encoder and save the embeddings.

The text encoder (~16GB bf16) is split across GPU and CPU so it runs in full
precision; every later run reuses these embeddings and never loads it again.

    python encode_prompts.py --prompt "a cat" --out embeds/smoke.pt
    python encode_prompts.py --prompts-file prompts.txt --out embeds/hpsv3_200.pt
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import Qwen3ForCausalLM

from common import BASE, DTYPE, NEG_PROMPT, asym_scheduler, limit_vram, oklab_encoder


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--prompt')
    src.add_argument('--prompts-file')
    ap.add_argument('--out', required=True)
    ap.add_argument('--neg-prompt', default=NEG_PROMPT)
    ap.add_argument('--gpu-mem', default='11GiB', help='GPU share of the text encoder weights')
    ap.add_argument('--max-vram-gb', type=float, default=13.0)
    args = ap.parse_args()
    limit_vram(args.max_vram_gb)

    if args.prompt:
        prompts = [args.prompt]
    else:
        lines = [l for l in Path(args.prompts_file).read_text().splitlines() if l.strip()]
        prompts = ([json.loads(l)['prompt'] for l in lines]
                   if args.prompts_file.endswith('.jsonl') else [l.strip() for l in lines])

    from lakonlab.pipelines.pipeline_pixelflux2_klein import PixelFlux2KleinPipeline
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        BASE, subfolder='text_encoder', torch_dtype=DTYPE,
        device_map='auto', max_memory={0: args.gpu_mem, 'cpu': '26GiB'})
    pipe = PixelFlux2KleinPipeline.from_pretrained(
        BASE, transformer=None, text_encoder=text_encoder,
        vae=oklab_encoder(), scheduler=asym_scheduler(), torch_dtype=DTYPE)

    def encode(text):
        embeds, _ = pipe.encode_prompt(prompt=text, device='cuda', num_images_per_prompt=1)
        return embeds.to('cpu')

    with torch.no_grad():
        neg = encode(args.neg_prompt)
        pos = []
        for i, p in enumerate(prompts):
            pos.append(encode(p))
            print(f'[{i + 1}/{len(prompts)}] {tuple(pos[-1].shape)} {p[:60]}', flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'prompts': prompts, 'prompt_embeds': pos,
                'neg_prompt': args.neg_prompt, 'neg_embeds': neg}, args.out)
    print(f'saved {len(pos)} prompt embeddings to {args.out}')


if __name__ == '__main__':
    main()
