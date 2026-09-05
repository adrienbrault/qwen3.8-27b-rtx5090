#!/usr/bin/env python3
"""Usage: CUDA_VISIBLE_DEVICES=0 python recalibrate_drafter.py --stage capture --target /target --corpus calib.jsonl --out /features.

Capture/preflight only. Quantization fails explicitly: no audited DFlash2
llm-compressor forward/export adapter is present in the supplied source dump.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct

LAYERS = (5, 19, 33, 47, 61)
REFERENCE = Path(__file__).with_name('drafter-reference.json')


def parameter_ledger(config):
    h, i, d = (config[k] for k in ('hidden_size', 'intermediate_size', 'head_dim'))
    q, kv, n, v = (config[k] for k in ('num_attention_heads', 'num_key_value_heads', 'num_hidden_layers', 'vocab_size'))
    dc = config['dflash_config']
    taps, groups, rank = dc['conv_kernel_size'], h // dc['conv_group_size'], dc['selector_rank']
    parts = dict(attention=n * (h * (q + 2 * kv) * d + q * d * h), mlp=n * 3 * h * i,
                 norms=n * (2 * h + 2 * d) + 2 * h,
                 convs=n * 2 * (h * 2 * taps * groups + 2 * taps * h),
                 fc=len(dc['target_layer_ids']) * h * h, selector=2 * v * rank + h * rank)
    core = sum(parts.values())
    return dict(parts=parts, core_parameters=core, embedding_parameters=v * h,
                head_parameters=v * h, core_bf16_GiB=core * 2 / 2**30,
                full_parameters=core + 2 * v * h,
                features_bytes_per_token=len(LAYERS) * h * 2)


def weight_headers(directory):
    """Inspect safe tensor metadata without loading tensor bodies or importing torch."""
    tensors = {}
    files = sorted(Path(directory).glob('*.safetensors'))
    if not files:
        raise ValueError('Local BF16 drafter needs .safetensors files')
    for file in files:
        with file.open('rb') as f:
            length_bytes = f.read(8)
            if len(length_bytes) != 8:
                raise ValueError(f'Truncated header: {file}')
            length = struct.unpack('<Q', length_bytes)[0]
            if length > 100_000_000:
                raise ValueError(f'Oversized header: {file}')
            header = json.loads(f.read(length))
        for name, spec in header.items():
            if name == '__metadata__':
                continue
            if name in tensors:
                raise ValueError(f'Duplicate tensor: {name}')
            tensors[name] = spec
    if any(x in name for name in tensors for x in ('weight_packed', 'weight_scale')):
        raise ValueError('Input is quantized; an original BF16 checkpoint is required')
    if any(spec['dtype'] != 'BF16' for name, spec in tensors.items() if name.endswith('weight')):
        raise ValueError('Expected BF16 weights')
    return dict(tensor_count=len(tensors), parameters=sum(math.prod(s['shape']) for s in tensors.values()),
                has_embedding=any('embed_tokens' in k for k in tensors),
                has_head=any('lm_head' in k for k in tensors))


def install_capture(model):
    """Runs inside the eager TP1 vLLM worker via LLM.apply_model."""
    import torch
    from vllm.logger import init_logger
    holders = [m for m in model.modules() if type(m).__name__ == 'Qwen3_5Model']
    if len(holders) != 1:
        raise RuntimeError('Expected one Qwen3_5Model')
    inner = holders[0]
    if len(inner.layers) != 64 or hasattr(model, '_r190_capture'):
        raise RuntimeError('Unexpected layer count or duplicate capture installation')
    state = dict(armed=False, chunks=[], features={}, positions=None)
    model._r190_capture = state

    def hook_for(layer_id):
        def hook(module, args, kwargs, output):
            if not state['armed']:
                return
            positions = kwargs.get('positions')
            if positions is not None and positions.ndim == 2 and positions.shape[0] == 3:
                if not torch.equal(positions[0], positions[1]) or not torch.equal(positions[0], positions[2]):
                    raise RuntimeError('Multimodal positions are unsupported')
                positions = positions[0]
            if positions is None or positions.ndim != 1:
                raise RuntimeError('Expected text-only positions in eager layer kwargs')
            hidden, residual = output
            feature = hidden + residual if residual is not None else hidden
            if feature.ndim != 2 or feature.shape[1] != 5120:
                raise RuntimeError('Unexpected target feature shape')
            pos = positions.detach().cpu().clone()
            if layer_id == LAYERS[0]:
                if state['features']:
                    raise RuntimeError('Incomplete previous feature group')
                state['positions'] = pos
            if not torch.equal(pos, state['positions']):
                raise RuntimeError('Layer position mismatch')
            state['features'][layer_id] = feature.detach().to(device='cpu', dtype=torch.bfloat16).clone()
            if layer_id == LAYERS[-1]:
                features = torch.cat([state['features'][i] for i in LAYERS], dim=-1)
                chunk = len(state['chunks'])
                path = Path(state['directory']) / f"{state['sample_id']}.{chunk:04d}.pt"
                torch.save(dict(positions=pos, features=features), path)
                state['chunks'].append(dict(file=path.name, positions=pos.tolist()))
                state['features'] = {}
        return hook

    state['handles'] = [inner.layers[i].register_forward_hook(hook_for(i), with_kwargs=True) for i in LAYERS]
    init_logger(__name__).info_once('R190-28 CAPTURE active TP1 eager post-layer=[5,19,33,47,61] width=25600')
    return True


def arm(model, directory, sample_id):
    s = model._r190_capture
    if s['armed']:
        raise RuntimeError('Capture already armed')
    s.update(armed=True, directory=directory, sample_id=sample_id, chunks=[], features={}, positions=None)
    return True


def finish(model, count):
    s = model._r190_capture
    s['armed'] = False
    positions = [p for chunk in s['chunks'] for p in chunk['positions']]
    if s['features'] or positions != list(range(count)):
        raise RuntimeError('Missing, duplicated, padded or reordered token rows; capture invalid')
    return s['chunks']


def capture(a):
    from functools import partial
    import vllm
    from vllm import LLM, SamplingParams
    if not vllm.__version__.startswith('0.29'):
        raise RuntimeError('Requires supplied vLLM 0.29 patched image')
    if os.environ.get('VLLM_SM12X_PCIE_IPC_AR', '0') != '0':
        raise RuntimeError('Disable TP2 pcie_ipc for this isolated TP1 process')
    if not Path(a.target).is_dir():
        raise ValueError('--target must be a local checkpoint directory')
    rows = [json.loads(x) for x in Path(a.corpus).read_text().splitlines() if x.strip()]
    if not rows or any(r.get('dry_run') for r in rows):
        raise ValueError('Need a non-test corpus')
    output = Path(a.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    llm = LLM(model=a.target, tensor_parallel_size=1, enforce_eager=True,
              enable_prefix_caching=False, max_num_seqs=1, max_model_len=4112,
              max_num_batched_tokens=4096, gpu_memory_utilization=0.88,
              kv_cache_memory_bytes=512 * 1024**2, kv_cache_dtype='nvfp4',
              mamba_ssm_cache_dtype='bfloat16', dtype='bfloat16',
              limit_mm_per_prompt={'image': 0, 'video': 0})
    llm.apply_model(install_capture)
    tokenizer = llm.get_tokenizer()
    with (output / 'manifest.jsonl').open('x') as manifest:
        for row in rows:
            ids = tokenizer.encode(row['text'], add_special_tokens=False)
            digest = hashlib.sha256(row['text'].encode()).hexdigest()
            if digest != row['id'] or not 1 <= len(ids) <= 4096 or len(ids) != row['num_tokens']:
                raise ValueError('Corpus hash/tokenizer/length mismatch')
            llm.apply_model(partial(arm, directory=str(output), sample_id=digest))
            llm.generate([{'prompt_token_ids': ids}], SamplingParams(max_tokens=1, temperature=0), use_tqdm=False)
            result = llm.apply_model(partial(finish, count=len(ids)))
            if len(result) != 1:
                raise RuntimeError('Capture must use TP1')
            record = dict(id=digest, kind=row['kind'], token_ids=ids, target=str(Path(a.target).resolve()),
                          layers=LAYERS, chunks=result[0], vllm_version=vllm.__version__,
                          corpus_sha256=hashlib.sha256(Path(a.corpus).read_bytes()).hexdigest(),
                          target_config_sha256=hashlib.sha256((Path(a.target) / 'config.json').read_bytes()).hexdigest(),
                          mode='TP1-eager-prefill-nvfp4-kv')
            manifest.write(json.dumps(record) + '\n')
            manifest.flush()
    print('R190-28 CAPTURE complete; feature cache only, NOT a recalibrated checkpoint')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('preflight', 'capture', 'quantize'), default='preflight')
    p.add_argument('--bf16-drafter')
    p.add_argument('--target')
    p.add_argument('--corpus')
    p.add_argument('--out')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    reference = json.loads(REFERENCE.read_text())
    ledger = parameter_ledger(reference)
    if a.dry_run:
        from build_calib_corpus import build, inline_sources, DryTokenizer
        assert len(build(inline_sources(), DryTokenizer(), 3, 1, 1024, 28)) == 3
        assert ledger['core_parameters'] == 1924404480
        assert ledger['features_bytes_per_token'] == 51200
        assert reference['quantization_config']['config_groups']['group_0']['weights']['group_size'] == 128
        print('R190-28 RECALIBRATION dry-run PASS (capture math only; quantization unavailable)')
        return
    if a.stage == 'quantize':
        p.exit(2, 'BLOCKED: no audited DFlash2 llm-compressor forward/export adapter in source dump. '
                  'Cannot re-quantize or emit a loadable checkpoint. See NOTES28.md.\n')
    if a.stage == 'capture':
        if not all((a.target, a.corpus, a.out)):
            p.error('capture needs --target, --corpus and a NEW --out directory')
        capture(a)
        return
    if not a.bf16_drafter:
        p.error('preflight needs --bf16-drafter')
    config = json.loads((Path(a.bf16_drafter) / 'config.json').read_text())
    for key in ('architectures', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
                'num_attention_heads', 'num_key_value_heads', 'head_dim', 'vocab_size', 'dflash_config'):
        if config.get(key) != reference[key]:
            raise ValueError(f'Checkpoint differs from served drafter: {key}')
    if config.get('quantization_config'):
        raise ValueError('Input must be original unquantized BF16 drafter')
    print(json.dumps(dict(status='preflight_only_quantization_blocked', ledger=ledger,
                          checkpoint=weight_headers(a.bf16_drafter),
                          required_output_scheme=reference['quantization_config']), indent=2))


if __name__ == '__main__':
    main()
