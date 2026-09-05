#!/usr/bin/env python3
# Usage (inside container): python nvfp4_gemm_census.py --json census.json --mode graph
"""Synthetic post-CT-loader weights; no engine, checkpoint, or distributed group."""
import argparse
import gc
import importlib.metadata
import json
import statistics

MS = [1, 2, 4, 8, 10, 16, 20, 32, 40, 64, 80, 128, 160, 256, 2048, 8192]
# Actual fused serving shapes plus explicitly requested component measurements.
SHAPES = {
    'gate_up': (17408, 5120, 'nvfp4'), 'down': (5120, 8704, 'nvfp4'),
    'qkv': (7168, 5120, 'fp8'), 'qkv_ungated': (4096, 5120, 'fp8'),
    'o_proj': (5120, 3072, 'fp8'), 'in_proj_qkv': (5120, 5120, 'fp8'),
    'in_proj_z': (3072, 5120, 'fp8'), 'in_proj_qkvz': (8192, 5120, 'fp8'),
    'out_proj': (5120, 3072, 'fp8'), 'lm_head': (124160, 5120, 'fp8'),
    'gate_up_fp8': (17408, 5120, 'fp8'), 'down_fp8': (5120, 8704, 'fp8'),
}


def weight_bytes(n, k, scheme):
    # Packed payload + local scales; scalar metadata separately negligible.
    return n * k // 2 + n * k // 16 if scheme == 'nvfp4' else n * k + 4 * n


def build_layer(torch, n, k, scheme):
    layer = torch.nn.Module()
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    layer.params_dtype = torch.bfloat16
    layer.orig_dtype = torch.bfloat16
    layer.logical_widths = [n // 2, n // 2] if n == 17408 else [n]
    layer.output_partition_sizes = layer.logical_widths
    layer.has_bias = False
    layer.weight_block_size = None
    def param(name, value):
        setattr(layer, name, torch.nn.Parameter(value, requires_grad=False))
    if scheme == 'nvfp4':
        # CT standardized layout AFTER divisor inversion / packed-name rename,
        # BEFORE kernel-specific postprocessing. All 16 E2M1 nibble codes finite.
        param('weight', torch.randint(0, 256, (n, k // 2), device='cuda', dtype=torch.uint8))
        param('weight_scale', (torch.rand(n, k // 16, device='cuda') + .5).to(torch.float8_e4m3fn))
        for name in ('weight_global_scale', 'input_global_scale', 'input_global_scale_inv', 'alpha'):
            param(name, torch.ones((), device='cuda', dtype=torch.float32))
    else:
        # CompressedTensorsW8A8Fp8 channel strategy transposes (N,K) to (K,N).
        param('weight', torch.randn(n, k, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn).t())
        layer.weight.input_dim, layer.weight.output_dim = 0, 1
        param('weight_scale', torch.full((n, 1), .125, device='cuda', dtype=torch.float32))
        layer.input_scale = None
        layer.input_scale_ub = None
    return layer


def measure(torch, kernel, layer, m, k, iters, warmup, mode):
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16) * .1
    def run():
        return kernel.apply_weights(layer, x, None)
    for _ in range(warmup):
        out = run()
    torch.cuda.synchronize()
    if not torch.isfinite(out).all().item():
        raise RuntimeError('nonfinite synthetic output')
    graph = None
    if mode == 'graph':
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
        run = graph.replay
        for _ in range(warmup):
            run()
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for _ in range(iters)]
    for start, end in pairs:
        start.record()
        run()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) * 1000 for a, b in pairs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shapes', default=','.join(SHAPES))
    p.add_argument('--kernels', default='all', help='comma-separated class names; FP8Auto selects served FP8 path')
    p.add_argument('--ms', default=','.join(map(str, MS)))
    p.add_argument('--iters', type=int, default=30)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--mode', choices=['graph', 'eager'], default='graph')
    p.add_argument('--json', default='census.json')
    args = p.parse_args()
    shapes = args.shapes.split(',')
    ms = [int(m) for m in args.ms.split(',')]
    if set(shapes) - SHAPES.keys() or not ms or min(ms) < 1 or args.iters < 20 or args.warmup < 1:
        p.error('unknown shape, invalid M, iters <20, or warmup <1')
    import torch
    import vllm
    from vllm.model_executor.kernels import linear
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym, kFp8StaticChannelSym
    from vllm.platforms import PlatformEnum
    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)
    versions = {'torch': torch.__version__, 'vllm': vllm.__version__,
                'flashinfer': importlib.metadata.version('flashinfer-python'),
                'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability()}
    if versions['capability'] != (12, 0):
        p.error('this census targets SM120')
    print(json.dumps(versions))
    registry = {c.__name__: c for c in linear._POSSIBLE_NVFP4_KERNELS[PlatformEnum.CUDA]}
    requested = set(args.kernels.split(','))
    if requested != {'all'} and requested - (registry.keys() | {'FP8Auto'}):
        p.error('unknown kernel name; use NVFP4 class names or FP8Auto')
    records, skipped = [], []
    baseline = type(linear.init_nvfp4_linear_kernel()).__name__
    print(f'NVFP4 ladder baseline: {baseline}; mode={args.mode}; weights reused (warm cache).')
    print('| shape | kernel | M | N | K | median us | effective TB/s | TFLOPS |')
    print('|---|---|---:|---:|---:|---:|---:|---:|')
    def skip(shape, name, reason, m=None):
        row = {'shape': shape, 'kernel': name, 'M': m, 'reason': str(reason)}
        skipped.append(row)
        print('SKIP/FAIL ' + json.dumps(row), flush=True)
    def save():
        with open(args.json, 'w') as f:
            json.dump({'versions': versions, 'mode': args.mode, 'iters': args.iters,
                       'baseline_nvfp4': baseline, 'records': records, 'skipped': skipped}, f, indent=2)
    with torch.inference_mode():
        for shape in shapes:
            n, k, scheme = SHAPES[shape]
            candidates = list(registry) if scheme == 'nvfp4' else ['FP8Auto']
            for name in candidates:
                if requested != {'all'} and name not in requested:
                    continue
                layer = kernel = None
                try:
                    if scheme == 'nvfp4':
                        cls = registry[name]
                        config = linear.NvFp4LinearLayerConfig()
                        for ok, reason in (cls.is_supported(), cls.can_implement(config)):
                            if not ok:
                                raise RuntimeError(reason)
                        kernel = cls(config)
                    else:
                        kernel = linear.init_fp8_linear_kernel(
                            weight_quant_key=kFp8StaticChannelSym, activation_quant_key=kFp8DynamicTokenSym,
                            input_dtype=torch.bfloat16, out_dtype=torch.bfloat16,
                            weight_shape=(n, k))
                    torch.manual_seed(23)  # identical raw weights across kernels
                    layer = build_layer(torch, n, k, scheme)
                    kernel.process_weights_after_loading(layer)
                    for m in ms:
                        try:
                            us = measure(torch, kernel, layer, m, k, args.iters, args.warmup, args.mode)
                            row = dict(shape=shape, kernel=type(kernel).__name__, M=m, N=n, K=k,
                                       scheme=scheme, us=us, weight_bytes=weight_bytes(n, k, scheme),
                                       TBps=weight_bytes(n, k, scheme)/us/1e6, TFLOPS=2*m*n*k/us/1e6)
                            records.append(row)
                            print(f'| {shape} | {row["kernel"]} | {m} | {n} | {k} | {us:.3f} | {row["TBps"]:.3f} | {row["TFLOPS"]:.3f} |', flush=True)
                        except Exception as exc:
                            skip(shape, name, f'{type(exc).__name__}: {exc}', m)
                        save()
                except Exception as exc:
                    skip(shape, name, f'{type(exc).__name__}: {exc}')
                finally:
                    del layer, kernel
                    gc.collect()
                    torch.cuda.empty_cache()
                    save()
    if not records:
        raise SystemExit('No timings produced; see skip reasons')


if __name__ == '__main__':
    main()
