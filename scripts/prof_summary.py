#!/usr/bin/env python3
"""Aggregate a torch-profiler chrome trace (json or json.gz) by CUDA kernel name.

  python3 prof_summary.py trace.json.gz [--top 40] [--steps N] [--label x]

Prints total GPU kernel time, the wall span covered, and the top kernels by total time with
count and mean. --steps N divides totals by N to give per-decode-step numbers (N = number of
engine iterations captured, e.g. the profiler's max_iterations). Families group kernels by role
so two arms can be compared at a glance (attention / gdn / gemm / other).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import sys

FAMILIES = [
    ("attn-fa2-prefill", re.compile(r"BatchPrefillWithPagedKVCache|BatchPrefillWithRaggedKV|SinglePrefill")),
    ("attn-fa2-decode", re.compile(r"BatchDecodeWithPagedKVCache")),
    ("attn-xqa", re.compile(r"xqa|XQA|kernel_mha")),
    ("attn-merge", re.compile(r"MergeState|VariableLengthMerge")),
    ("kv-store", re.compile(r"reshape_and_cache|nvfp4_kv_cache")),
    ("gdn", re.compile(r"fused_recurrent|chunk_|gated_delta|causal_conv1d|_fwd_kernel|gdn|GatedDelta", re.I)),
    ("gemm", re.compile(r"gemm|Gemm|cutlass|nvfp4|fp4|Fp4|scaled_mm|matmul|cublas|Cutlass|sm120|marlin|Marlin|awq|gptq", re.I)),
    ("nccl", re.compile(r"nccl|ncclDev|AllReduce|allreduce", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized|reduce_kernel|rms_norm|RMSNorm|silu|rotary|Rotary|index_|gather|scatter|copy_|fill|cat|Cat|softmax|topk|Topk|sort|Sort|argmax", re.I)),
]


def family(name: str) -> str:
    for fam, rx in FAMILIES:
        if rx.search(name):
            return fam
    return "other"


def short(name: str, n: int = 110) -> str:
    name = re.sub(r"\(.*", "", name)  # drop template args in parens
    name = name.replace("void ", "")
    return name if len(name) <= n else name[: n - 3] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--steps", type=float, default=0.0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    opener = gzip.open if args.trace.endswith(".gz") else open
    with opener(args.trace, "rt") as f:
        data = json.load(f)
    ev = data["traceEvents"] if isinstance(data, dict) else data
    by_name = collections.defaultdict(lambda: [0.0, 0])
    t_min, t_max, total = float("inf"), 0.0, 0.0
    n_kernels = 0
    graphs = 0
    for e in ev:
        cat = e.get("cat", "")
        if cat == "kernel" or cat == "gpu_memcpy" or cat == "gpu_memset":
            d = float(e.get("dur", 0.0))
            nm = e["name"] if cat == "kernel" else f"[{cat}]"
            by_name[nm][0] += d
            by_name[nm][1] += 1
            total += d
            n_kernels += 1
            ts = float(e.get("ts", 0.0))
            t_min = min(t_min, ts)
            t_max = max(t_max, ts + d)
        elif cat == "cuda_runtime" and "GraphLaunch" in e.get("name", ""):
            graphs += 1
    span = (t_max - t_min) if n_kernels else 0.0
    steps = args.steps or 1.0
    unit = "ms/step" if args.steps else "ms"
    print(f"== {args.label or args.trace}")
    print(f"kernels={n_kernels} graph_launches={graphs} gpu_busy={total/1000:.1f} ms wall_span={span/1000:.1f} ms "
          f"busy/span={total/max(span,1):.2f}" + (f"  steps={args.steps:g} -> busy {total/1000/steps:.2f} ms/step, span {span/1000/steps:.2f} ms/step" if args.steps else ""))
    fams = collections.defaultdict(lambda: [0.0, 0])
    for nm, (d, c) in by_name.items():
        fams[family(nm)][0] += d
        fams[family(nm)][1] += c
    print("families (" + unit + "):")
    for fam, (d, c) in sorted(fams.items(), key=lambda kv: -kv[1][0]):
        print(f"  {fam:<18} {d/1000/steps:>9.3f}  {100*d/max(total,1):>5.1f}%  n={c/steps:g}")
    print(f"top {args.top} kernels ({unit}, count/step, mean us):")
    for nm, (d, c) in sorted(by_name.items(), key=lambda kv: -kv[1][0])[: args.top]:
        print(f"  {d/1000/steps:>9.3f} {c/steps:>8g} {d/c:>9.1f}  [{family(nm)}] {short(nm)}")


if __name__ == "__main__":
    sys.exit(main())
