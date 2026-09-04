#!/usr/bin/env python3
"""Decode-only step breakdown from a vLLM torch trace (R183). The benchy recipe puts 2,048-token prefill steps inside the 60-step
window; their all-reduces (21 MB > the 8 MiB custom-AR cap) run on NCCL and dominate naive per-step tables. This takes the window
AFTER the last NCCL all-reduce (pure decode: graph replays), counts decode steps as GDN-update kernels / 48 layers, and sums GPU
time per category per step. Usage: prof_decode_split.py trace.json.gz [--gdn-layers 48]"""
import gzip, json, sys, re, collections
p = sys.argv[1]; L = 48
d = json.load(gzip.open(p, "rt")); ev = d.get("traceEvents", d)
k = sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda e: e["ts"])
nccl = [e for e in k if "ncclDevKernel_AllReduce" in e["name"]]
t0 = (nccl[-1]["ts"] + nccl[-1]["dur"]) if nccl else k[0]["ts"]
w = [e for e in k if e["ts"] >= t0]
steps = sum(1 for e in w if e["name"].startswith("fused_sigmoid_gating_delta_rule_update_kernel")) / L
span = (w[-1]["ts"] + w[-1]["dur"] - w[0]["ts"]) / 1000.0
def cat(n):
    if "cross_device_reduce" in n: return "custom all-reduce"
    if "ncclDevKernel_AllReduce" in n: return "nccl all-reduce"
    if "ncclDevKernel" in n: return "nccl all-gather"
    if "marlin::Marlin" in n or "wmma_tensorop_bf16" in n: return "drafter GEMM (marlin/wmma)"
    if "cutlass" in n and ("Gemm" in n or "gemm" in n): return "target NVFP4 GEMM (cutlass)"
    if "fused_sigmoid_gating" in n or "causal_conv1d" in n or "chunk_" in n or "gated_delta" in n or "merge_16x16" in n: return "GDN kernels"
    if "flashinfer" in n or "attention" in n.lower(): return "attention (flashinfer)"
    if "topk_topp" in n or "sampl" in n.lower(): return "sampler"
    if "cvt_fp16_to_fp4" in n or "fp4_quant" in n: return "fp4 quant (act)"
    if n.startswith("triton_"): return "triton fused (norm/quant/misc)"
    if "elementwise" in n or "vectorized" in n or "reduce_kernel" in n or "index" in n.lower() or "copy" in n.lower(): return "elementwise/index/copy"
    if n.startswith("Memcpy") or n.startswith("Memset"): return "memcpy/memset"
    return "other"
tot = collections.defaultdict(float); cnt = collections.Counter()
for e in w: c = cat(e["name"]); tot[c] += e["dur"]; cnt[c] += 1
busy = sum(tot.values()) / 1000.0
print(f"{p.split('/')[-1][:5]}: decode window after last NCCL AR: span {span:.1f} ms, decode steps {steps:.1f}, span/step {span/steps:.2f} ms, GPU busy/step {busy/steps:.2f} ms, idle {100*(1-busy/span):.1f}%")
for c, v in sorted(tot.items(), key=lambda x: -x[1]):
    print(f"   {v/1000/steps:7.2f} ms/step  {cnt[c]/steps:7.1f} calls/step  {c}")
