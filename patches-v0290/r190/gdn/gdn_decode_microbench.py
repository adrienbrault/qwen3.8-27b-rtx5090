#!/usr/bin/env python3
# Usage: docker exec <container> python /path/gdn_decode_microbench.py --device 0
"""Single-token packed GDN sweep; rows 10/80/160 are NOT spec sequences."""
import argparse
import os
import statistics


def main():
    import torch
    from vllm import envs
    from vllm.third_party.flash_linear_attention.ops import fused_recurrent as fr

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--baseline-atol', type=float, default=0.03125)
    p.add_argument('--reference-atol', type=float, default=0.05)
    args = p.parse_args()
    if args.repeats < 3:
        p.error('--repeats must be >=3')
    torch.cuda.set_device(args.device)
    torch.manual_seed(141)
    envs.disable_envs_cache()
    os.environ.pop('VLLM_SM12X_GDN_PACKED_BV', None)
    os.environ.pop('VLLM_SM12X_GDN_DECODE_CFG', None)
    configs = [None]
    # 12 candidates: two algorithms/grid orders, 3 tile sizes, 2 warp counts.
    for variant in ('tiled', 'split'):
        for bv in (16, 32, 64):
            for warps in (1, 2):
                split = 1 if variant == 'tiled' else 2
                configs.append(f'bv={bv},bk=128,warps={warps},stages=3,'
                               f'split={split},variant={variant}')
    configs += ['bv=32,bk=128,warps=4,stages=1,split=2,variant=split',
                'bv=32,bk=128,warps=2,stages=2,split=4,variant=split']
    device = torch.device('cuda', args.device)
    failed = False
    print('Rows are independent single-token requests; graph replay kernel-only median;')
    print('GB/s = logical state read+write only (cache-warm, not measured DRAM bandwidth).')
    print('Columns: rows config us GB/s out/base state/base out/fp32 state/fp32 bitwise')
    for rows in (1, 8, 16, 10, 80, 160):
        def rand(*shape):
            return torch.randn(*shape, device=device, dtype=torch.bfloat16)
        mixed = rand(rows, 5120)
        a, b = rand(rows, 24), rand(rows, 24)
        A_log = torch.randn(24, device=device) * 0.1
        bias = torch.randn(24, device=device) * 0.1
        seed = rand(rows + 1, 24, 128, 128) * 0.1
        state = seed.clone()
        indices = torch.arange(rows, 0, -1, device=device, dtype=torch.int32)
        out = torch.empty(rows, 1, 24, 128, device=device, dtype=torch.bfloat16)
        scale = 128 ** -0.5
        # Pure torch fp32 reference, no bf16 cast until comparison. Group ratio 3.
        q = mixed[:, :1024].float().reshape(rows, 8, 128).repeat_interleave(3, 1)
        k = mixed[:, 1024:2048].float().reshape(rows, 8, 128).repeat_interleave(3, 1)
        v = mixed[:, 2048:].float().reshape(rows, 24, 128)
        q = q / torch.sqrt((q*q).sum(-1, keepdim=True) + 1e-6) * scale
        k = k / torch.sqrt((k*k).sum(-1, keepdim=True) + 1e-6)
        x = a.float() + bias
        softplus = torch.where(x <= 20, torch.log(1 + torch.exp(x)), x)
        decay = torch.exp(-torch.exp(A_log) * softplus)
        ref_h = seed[indices.long()].float() * decay[..., None, None]
        delta = (v - (ref_h * k[..., None, :]).sum(-1)) * torch.sigmoid(b.float())
        ref_h = ref_h + delta[..., None] * k[..., None, :]
        ref_o = (ref_h * q[..., None, :]).sum(-1).unsqueeze(1)
        base_o = base_h = None
        for cfg in configs:
            if cfg is None:
                os.environ.pop('VLLM_SM12X_GDN_DECODE_CFG', None)
            else:
                os.environ['VLLM_SM12X_GDN_DECODE_CFG'] = cfg
            fr._get_packed_decode_launch_config.cache_clear()
            def launch():
                fr.fused_recurrent_gated_delta_rule_packed_decode(
                    mixed, a, b, A_log, bias, scale, state, out, indices, True)
            # Compilation/unsupported configurations are errors, never skipped.
            state.copy_(seed)
            launch()
            torch.cuda.synchronize()
            if cfg is None:
                base_o, base_h = out.clone(), state.clone()
            def diff(lhs, rhs):
                return (lhs.float() - rhs.float()).abs().max().item()
            errors = (diff(out, base_o), diff(state, base_h),
                      diff(out, ref_o), diff(state[indices.long()], ref_h))
            exact = torch.equal(out, base_o) and torch.equal(state, base_h)
            ok = all(e <= args.baseline_atol for e in errors[:2]) and all(
                e <= args.reference_atol for e in errors[2:])
            # NULL slot produces zero output and preserves all state bytes.
            saved = indices[-1].item()
            indices[-1] = 0
            state.copy_(seed)
            launch()
            assert torch.count_nonzero(out[-1]).item() == 0
            assert torch.equal(state[0], seed[0])
            assert torch.equal(state[saved], seed[saved])
            indices[-1] = saved
            # Warm up on a side stream before capture. State reset is NOT timed.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    state.copy_(seed)
                    launch()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                launch()
            timings = []
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(args.repeats):
                state.copy_(seed)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end) * 1000)
            # Replay equality to eager catches stale addresses / capture mistakes.
            assert diff(out, base_o) <= args.baseline_atol
            assert diff(state, base_h) <= args.baseline_atol
            us = statistics.median(timings)
            bandwidth = rows * 24 * 128 * 128 * 2 * 2 / (us * 1000)
            print(rows, cfg or 'baseline(bv32,w1,s3)', f'{us:.3f}', f'{bandwidth:.2f}',
                  *(f'{e:.6g}' for e in errors), exact, 'PASS' if ok else 'FAIL', flush=True)
            failed |= not ok
    if failed:
        raise SystemExit('Numerical tolerance exceeded; do not promote.')


if __name__ == '__main__':
    main()
