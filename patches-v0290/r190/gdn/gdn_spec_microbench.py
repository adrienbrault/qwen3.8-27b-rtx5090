#!/usr/bin/env python3
# Usage: docker exec <container> python /opt/gdn_spec_microbench.py --device 0
"""ns9 GDN: eager/event and single-kernel CUDA-graph replay sweep, one GPU."""
import argparse
import math
import os
import statistics


def reference(torch, q, k, v, a, b, alog, bias, seed, indices, accepted, n):
    # Independent fp32 recurrence; bf16 checkpoints are NOT fed into the loop.
    q = q.float().reshape(n, 10, 8, 128).repeat_interleave(3, 2)
    k = k.float().reshape(n, 10, 8, 128).repeat_interleave(3, 2)
    q *= torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * 128**-0.5
    k *= torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    v = v.float().reshape(n, 10, 24, 128)
    x = a.float().reshape(n, 10, 24) + bias.float()
    sp = torch.where(x <= 20, torch.log(1 + torch.exp(x)), x)
    decay = torch.exp(-torch.exp(alog.float()) * sp)
    gate = torch.sigmoid(b.float().reshape(n, 10, 24))
    h = seed[indices[torch.arange(n, device=q.device), accepted.long()-1].long()].float()
    states, outputs = [], []
    for t in range(10):
        h = h * decay[:, t, :, None, None]
        delta = (v[:, t] - (h * k[:, t, :, None, :]).sum(-1)) * gate[:, t, :, None]
        h = h + delta[..., None] * k[:, t, :, None, :]
        outputs.append((h * q[:, t, :, None, :]).sum(-1))
        states.append(h.clone())
    return torch.stack(outputs, 1).reshape(1, n*10, 24, 128), torch.stack(states, 1)


def main():
    import torch
    from vllm import envs
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update as update,
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--baseline-atol', type=float, default=0.00390625)
    p.add_argument('--reference-atol', type=float, default=0.00390625)
    p.add_argument('--reverse', action='store_true', help='reverse candidate order')
    args = p.parse_args()
    if args.repeats < 3 or any(not math.isfinite(x) or x < 0 for x in
                             (args.baseline_atol, args.reference_atol)):
        p.error('repeats>=3 and finite nonnegative tolerances required')
    torch.cuda.set_device(args.device)
    torch.manual_seed(145)
    envs.disable_envs_cache()
    knob = 'VLLM_SM12X_GDN_SPEC_CFG'
    candidates = []
    for variant in ('tiled', 'split'):
        for bv, warps, stages in ((8, 1, 3), (16, 1, 3), (16, 2, 3),
                                  (32, 2, 3), (32, 4, 3), (16, 2, 1),
                                  (16, 2, 2), (32, 2, 1), (32, 2, 2)):
            split = 1 if variant == 'tiled' else (32 // bv)
            nominal = bv * split
            candidates.append(f'bv={nominal},bk=128,warps={warps},stages={stages},'
                              f'split={split},variant={variant}')
    if args.reverse:
        candidates.reverse()
    results = {}
    failed = False
    print('us: eager CUDA events include Python launch gaps; graph=single replay events.')
    print('Reset copies outside timing; cache-warm synthetic states; no model traffic.')
    print('N config eager_us graph_us speedup errors[out/base,allstate/base,out/fp32,'
          'checkpoint/fp32,final/fp32] bitwise replay_bitwise status', flush=True)
    for n in (1, 8, 16):
        device = torch.device('cuda', args.device)
        def rand(*shape):
            return torch.randn(*shape, device=device, dtype=torch.bfloat16)
        q, k = rand(1, n*10, 8, 128), rand(1, n*10, 8, 128)
        v, a, b = rand(1, n*10, 24, 128), rand(n*10, 24), rand(n*10, 24)
        alog = torch.randn(24, device=device) * 0.1
        bias = torch.randn(24, device=device) * 0.1
        seed = rand(n*10+1, 24, 128, 128) * 0.1
        state = seed.clone()
        # Disjoint, permuted slots, plus untouched NULL sentinel 0.
        indices = (torch.randperm(n*10, device=device) + 1).int().reshape(n, 10)
        accepted = torch.randint(1, 11, (n,), device=device, dtype=torch.int32)
        cu = torch.arange(n+1, device=device, dtype=torch.int32) * 10
        ref_o, ref_h = reference(torch, q, k, v, a, b, alog, bias, seed,
                                 indices, accepted, n)
        def launch():
            return update(alog, a, b, bias, q, k, v, initial_state=state,
                          inplace_final_state=True, cu_seqlens=cu,
                          ssm_state_indices=indices, num_accepted_tokens=accepted,
                          use_qk_l2norm_in_kernel=True)[0]
        def diff(lhs, rhs):
            return (lhs.float() - rhs.float()).abs().max().item()
        base_o = base_h = None
        for cfg in [None] + candidates:
            if cfg is None:
                os.environ.pop(knob, None)
            else:
                os.environ[knob] = cfg
            state.copy_(seed)
            out = launch()  # First call compiles. Compile failures abort.
            torch.cuda.synchronize()
            eager_o, eager_h = out.clone(), state.clone()
            if cfg is None:
                base_o, base_h = eager_o, eager_h
            checks = state[indices.long()]
            errors = (diff(out, base_o), diff(state, base_h), diff(out, ref_o),
                      diff(checks, ref_h), diff(checks[:, -1], ref_h[:, -1]))
            exact = torch.equal(out, base_o) and torch.equal(state, base_h)
            ok = all(math.isfinite(e) and e <= args.baseline_atol for e in errors[:2])
            ok &= all(math.isfinite(e) and e <= args.reference_atol for e in errors[2:])
            assert torch.equal(state[0], seed[0]), 'NULL sentinel overwritten'
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    state.copy_(seed)
                    launch()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                graph_out = launch()
            # Validate replay on the original inputs AND a changed acceptance vector.
            for change in (False, True):
                if change:
                    accepted.copy_(11 - accepted)
                    state.copy_(seed)
                    changed_o = launch().clone()
                    changed_h = state.clone()
                state.copy_(seed)
                graph.replay()
                torch.cuda.synchronize()
                want_o, want_h = (changed_o, changed_h) if change else (eager_o, eager_h)
                assert torch.equal(graph_out, want_o) and torch.equal(state, want_h), \
                    'graph/eager mismatch'
                if change:
                    accepted.copy_(11 - accepted)
            times = []
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for graph_mode in (False, True):
                samples = []
                for _ in range(args.repeats):
                    state.copy_(seed)
                    start.record()
                    if graph_mode:
                        graph.replay()
                    else:
                        out = launch()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) * 1000)
                times.append(statistics.median(samples))
            results[n, cfg] = times[1]
            print(n, cfg or 'baseline(bv32,w4,s3)', *(f'{t:.3f}' for t in times),
                  f'{results[n, None]/times[1]:.3f}', *(f'{e:.6g}' for e in errors),
                  exact, True, 'PASS' if ok else 'FAIL', flush=True)
            failed |= not ok
            del graph, graph_out
    winners = [c for c in candidates if results[1, c] <= 0.9 * results[1, None]
               and results[8, c] <= results[8, None]
               and results[16, c] <= results[16, None]]
    print('>=10% c1 time reduction with no c8/c16 loss:', winners or 'NONE (dead)')
    if failed:
        raise SystemExit('Numerical gate failed; no candidate may be promoted from this run.')


if __name__ == '__main__':
    main()
