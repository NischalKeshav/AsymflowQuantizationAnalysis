"""Generate images from precomputed embeddings for one (model, quant) setting.

bf16 does not fit in 16GB VRAM, so it streams transformer blocks from CPU
(group offloading). Quantized settings load straight onto the GPU.

    python generate.py --model asym --quant bf16 --embeds embeds/smoke.pt \
        --out out/smoke --seed 42 --width 960 --height 1280
"""
import argparse
import json
import time
from pathlib import Path

import torch

from common import ADAPTER, BASE, DTYPE, GEN, asym_scheduler, limit_vram, oklab_encoder, quant_config


def build_pipeline(model, quant, resident=False):
    from diffusers import Flux2Transformer2DModel
    transformer = Flux2Transformer2DModel.from_pretrained(
        BASE, subfolder='transformer', torch_dtype=DTYPE,
        quantization_config=quant_config(quant))
    if model == 'asym':
        from lakonlab.pipelines.pipeline_pixelflux2_klein import PixelFlux2KleinPipeline
        pipe = PixelFlux2KleinPipeline.from_pretrained(
            BASE, transformer=transformer, text_encoder=None, tokenizer=None,
            vae=oklab_encoder(), scheduler=asym_scheduler(), torch_dtype=DTYPE)
        pipe.load_lakonlab_adapter(ADAPTER, target_module_name='transformer')
    else:
        from diffusers import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(
            BASE, transformer=transformer, text_encoder=None, tokenizer=None, torch_dtype=DTYPE)

    if quant == 'bf16' and not resident:
        # 16GB GPU: stream transformer blocks from RAM.
        pipe.transformer.enable_group_offload(
            onload_device=torch.device('cuda'), offload_device=torch.device('cpu'),
            offload_type='block_level', num_blocks_per_group=1, use_stream=True,
            # Pin each block only while it is being transferred instead of pinning the
            # whole 18GB model up front; that up-front copy is what pushed RAM past 30GB.
            low_cpu_mem_usage=True)
        for name in ('vae',):
            module = getattr(pipe, name, None)
            if isinstance(module, torch.nn.Module):
                module.to('cuda')
    else:
        pipe.to('cuda')
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['asym', 'latent'], required=True)
    ap.add_argument('--quant', default='bf16')
    ap.add_argument('--embeds', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=None,
                    help='fixed seed for every prompt; default is seed = prompt index')
    ap.add_argument('--start', type=int, default=0, help='first prompt index (resume / skip pilot images)')
    ap.add_argument('--limit', type=int, default=None, help='last prompt index, exclusive')
    ap.add_argument('--width', type=int, default=GEN['width'])
    ap.add_argument('--height', type=int, default=GEN['height'])
    ap.add_argument('--steps', type=int, default=GEN['num_inference_steps'])
    ap.add_argument('--max-vram-gb', type=float, default=13.0)
    ap.add_argument('--resident', action='store_true',
                    help='bf16 fully on the GPU (needs >=24GB, e.g. Colab A100/L4) instead of CPU offload')
    args = ap.parse_args()
    limit_vram(args.max_vram_gb)
    device_name = torch.cuda.get_device_name()

    # mmap: only the embedding in use is read into RAM, not the whole 2.5GB file.
    data = torch.load(args.embeds, mmap=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pipe = build_pipeline(args.model, args.quant, resident=args.resident)
    print(f'pipeline ready in {time.time() - t0:.0f}s', flush=True)

    n = len(data['prompt_embeds']) if args.limit is None else args.limit
    with open(out / 'log.jsonl', 'a') as log:
        for i in range(args.start, n):
            seed = args.seed if args.seed is not None else i
            if (out / f'{i:04d}_seed{seed}.png').exists():
                continue  # resume: skip images already generated
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t = time.time()
            image = pipe(
                prompt_embeds=data['prompt_embeds'][i].to('cuda', DTYPE),
                negative_prompt_embeds=data['neg_embeds'].to('cuda', DTYPE),
                width=args.width, height=args.height,
                num_inference_steps=args.steps, guidance_scale=GEN['guidance_scale'],
                generator=torch.Generator().manual_seed(seed),
            ).images[0]
            torch.cuda.synchronize()
            rec = dict(model=args.model, quant=args.quant, index=i, seed=seed,
                       seconds=round(time.time() - t, 2),
                       peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                       device=device_name)
            image.save(out / f'{i:04d}_seed{seed}.png')
            log.write(json.dumps(rec) + '\n')
            log.flush()
            print(rec, flush=True)


if __name__ == '__main__':
    main()
