"""Shared settings and pipeline builders for the quantization experiment."""
import math

import torch

BASE = 'black-forest-labs/FLUX.2-klein-base-9B'
ADAPTER = 'Lakonik/AsymFLUX.2-klein-9B'
# Benchmark settings match LakonLab configs/asymflow/asymflux2_klein_test.py (the paper's
# HPSv3 eval): empty negative prompt, 32 UniPC steps, guidance 4.0, 1024x1024.
NEG_PROMPT = ''
DEMO_NEG_PROMPT = 'Low quality, worst quality, blurry, deformed, bad anatomy, unclear text'
GEN = dict(width=1024, height=1024, num_inference_steps=32, guidance_scale=4.0)
DTYPE = torch.bfloat16


def limit_vram(max_gb, min_gb=4.0):
    """Cap this process's PyTorch VRAM so it can never take the whole GPU.

    The cap is the smaller of `max_gb` and what is free right now minus 0.5GB
    of headroom for the desktop. Going over raises an OOM in this process only.
    """
    free, total = torch.cuda.mem_get_info()
    budget = min(max_gb * 2 ** 30, free - 2 ** 29)
    if budget < min_gb * 2 ** 30:
        raise SystemExit(
            f'Only {free / 2 ** 30:.1f}GB of {total / 2 ** 30:.1f}GB VRAM is free. '
            'Close other GPU programs (e.g. unload the model in LM Studio) and retry.')
    torch.cuda.set_per_process_memory_fraction(budget / total)
    print(f'VRAM cap: {budget / 2 ** 30:.1f}GB '
          f'(free {free / 2 ** 30:.1f}GB of {total / 2 ** 30:.1f}GB)', flush=True)


def asym_scheduler():
    from lakonlab.models.diffusions.schedulers import FlowAdapterScheduler
    # Copied from the official AsymFlow example (docs/AsymFlow.md).
    return FlowAdapterScheduler(
        shift=17.0,
        use_dynamic_shifting=True,
        base_seq_len=1024 ** 2,
        max_seq_len=2048 ** 2,
        base_logshift=math.log(17.0),
        max_logshift=math.log(34.0),
        dynamic_shifting_type='sqrt',
        base_scheduler='UniPCMultistep')


def oklab_encoder():
    from lakonlab.models.architectures import OklabColorEncoder
    return OklabColorEncoder(use_affine_norm=True, mean=(0.56, 0.0, 0.01), std=0.16)


def quant_config(quant):
    """Diffusers quantization config for the transformer, or None for bf16."""
    if quant == 'bf16':
        return None
    if quant == 'nf4':
        from diffusers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=DTYPE)
    if quant == 'bnb8':
        from diffusers import BitsAndBytesConfig
        return BitsAndBytesConfig(load_in_8bit=True)
    # Any torchao quant type name (int8wo, int8dq, float8wo, float8dq, float8dq_e4m3_row, int4wo, ...).
    if quant.startswith(('int', 'uint', 'float')):
        from diffusers import TorchAoConfig
        return TorchAoConfig(quant)
    raise ValueError(f'unknown quant setting: {quant}')
