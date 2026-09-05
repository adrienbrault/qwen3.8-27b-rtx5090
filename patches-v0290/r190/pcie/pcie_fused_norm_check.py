#!/usr/bin/env python3
# Usage: torchrun --standalone --nproc-per-node=2 /opt/pcie_fused_norm_check.py [--atol 0]
"""Run inside the built container on two free GPUs. Default gate is bitwise."""
import argparse
import os
import torch
import torch.distributed as dist
from pcie_ipc_ar21.workspace import PcieIpcAllReduceWorkspace
from vllm.model_executor.layers.layernorm import GemmaRMSNorm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--atol', type=float, default=0.0)
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    assert dist.get_world_size() == 2
    group = dist.new_group(backend='gloo')
    device = torch.device('cuda', rank)
    ws = PcieIpcAllReduceWorkspace(group, device)
    shapes = [(m, 5120) for m in (1, 10, 80, 160, 2048)]
    ws.prepare(shapes)
    norm = GemmaRMSNorm(5120, eps=1e-6).to(device=device, dtype=torch.bfloat16)
    torch.manual_seed(144)
    with torch.no_grad():
        norm.weight.copy_(torch.randn_like(norm.weight) * .1)
    reference_norm = torch.compile(norm.forward_native, fullgraph=True)
    failures = 0
    try:
        # Ascending and descending sweeps exercise shared epochs across tactics/sizes.
        for m, h in shapes + list(reversed(shapes)):
            torch.manual_seed(144 + rank + m)
            x = torch.randn(m, h, device=device, dtype=torch.bfloat16)
            r = torch.randn_like(x)
            dist.broadcast(r, src=0)
            # Exercise signed zero sentinels too.
            x[:, ::23] = 0.0
            x[:, 1::23] = -0.0

            def ar():
                if m <= 320:
                    return ws.all_reduce(x)
                y = x.clone()
                dist.all_reduce(y)
                return y

            def unfused():
                return reference_norm(ar(), r)

            def fused():
                if m > 320:
                    return unfused()  # explicit served capacity fallback
                return ws.native.fused_norm(ws._handle, x, r, norm.weight, 1e-6)

            graphs = []
            outputs = []
            for fn in (unfused, fused):
                stream = torch.cuda.Stream(device=device)
                with torch.cuda.stream(stream), ws.capture():
                    for _ in range(3):
                        fn()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        output = fn()
                graphs.append(graph)
                outputs.append(output)
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            times = []
            for graph in graphs:
                dist.barrier()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iters):
                    graph.replay()
                end.record()
                end.synchronize()
                elapsed = torch.tensor(start.elapsed_time(end) * 1000 / args.iters, device=device)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                times.append(elapsed.item())
            errors = [(a.float() - b.float()).abs().max().item()
                      for a, b in zip(*outputs)]
            mismatches = [int((a.view(torch.int16) != b.view(torch.int16)).sum())
                          for a, b in zip(*outputs)]
            # At zero tolerance signed-bit equality is also required.
            bad = any(e > args.atol or not torch.isfinite(torch.tensor(e)) for e in errors)
            bad |= args.atol == 0 and any(mismatches)
            failures += int(bad)
            print(f'rank={rank} M={m} fallback={m > 320} max_abs(norm,res)={errors} '
                  f'bit_mismatches={mismatches} unfused_us={times[0]:.3f} '
                  f'fused_us={times[1]:.3f} saving_us={times[0]-times[1]:.3f}', flush=True)
        status = torch.tensor(failures, device=device)
        dist.all_reduce(status, op=dist.ReduceOp.MAX)
        failures = status.item()
    finally:
        ws.destroy()
        dist.destroy_process_group()
    if failures:
        raise SystemExit(f'FAIL: {failures} cases exceeded the requested numerical gate')
    print('PASS: graph comparison and timing complete', flush=True)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
