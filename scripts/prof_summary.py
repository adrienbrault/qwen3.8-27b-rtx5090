#!/usr/bin/env python3
"""Summarize torch-profiler chrome traces (vLLM --profiler-config.profiler=torch) into a kernel table.

R168 (2026-09-03): the rc1 nvfp4 route decodes 30K-context prompts at 29 tok/s vs 144 on v0.28; the kernel table of ~32
decode steps at 30K on each image is what attributes the missing 55 ms/step to a kernel (FlashInfer FA2 split-KV
decode, the overlay writer, GDN, the drafter) or to launch gaps (graph replay vs eager). Per input file (one per TP rank,
.pt.trace.json or .json.gz) it prints: wall span, number of CUDA kernels, cudaGraphLaunch count, and the top-N kernels by
total GPU time with count and mean; then a merged table across ranks. `--steps N` divides totals per step.
Usage: prof_summary.py <trace files or dir...> [--top 30] [--steps N] [--json out.json]
"""
import argparse, gzip, json, os, sys, collections


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    return d.get("traceEvents", d) if isinstance(d, dict) else d


def summarize(events):
    kern = collections.defaultdict(lambda: [0, 0.0])
    t0 = None; t1 = None; n_graph = 0; n_kern = 0; kern_time = 0.0
    for e in events:
        if e.get("ph") != "X":
            continue
        ts = e.get("ts", 0); dur = e.get("dur", 0)
        if t0 is None or ts < t0: t0 = ts
        if t1 is None or ts + dur > t1: t1 = ts + dur
        cat = e.get("cat", "")
        name = e.get("name", "")
        if cat == "kernel":
            k = kern[name]; k[0] += 1; k[1] += dur; n_kern += 1; kern_time += dur
        elif cat == "cuda_runtime" and "GraphLaunch" in name:
            n_graph += 1
    span = (t1 - t0) if (t0 is not None and t1 is not None) else 0.0
    return dict(span_ms=span / 1000.0, kernels=n_kern, kernel_time_ms=kern_time / 1000.0, graph_launches=n_graph,
                table={k: dict(count=v[0], total_ms=v[1] / 1000.0) for k, v in kern.items()})


def short(name, n=95):
    return name if len(name) <= n else name[: n - 3] + "..."


def print_table(tag, s, top, steps):
    print(f"== {tag}: span {s['span_ms']:.1f} ms, {s['kernels']} kernels totalling {s['kernel_time_ms']:.1f} ms GPU, "
          f"{s['graph_launches']} cudaGraphLaunch" + (f", per step: span {s['span_ms']/steps:.2f} ms, GPU {s['kernel_time_ms']/steps:.2f} ms, "
          f"{s['kernels']/steps:.0f} kernels, {s['graph_launches']/steps:.1f} graph launches" if steps else ""))
    rows = sorted(s["table"].items(), key=lambda kv: -kv[1]["total_ms"])[:top]
    print(f"{'total_ms':>9} {'count':>7} {'mean_us':>8} {'per-step':>9}  kernel")
    for name, r in rows:
        ps = f"{r['total_ms']/steps:.2f}" if steps else "-"
        print(f"{r['total_ms']:9.1f} {r['count']:7d} {1000*r['total_ms']/r['count']:8.1f} {ps:>9}  {short(name)}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("paths", nargs="+"); ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--steps", type=int, default=0); ap.add_argument("--json", default="")
    a = ap.parse_args()
    files = []
    for p in a.paths:
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in sorted(os.listdir(p)) if ".trace.json" in f or f.endswith(".json") or f.endswith(".json.gz")]
        else:
            files.append(p)
    if not files:
        print("no trace files", file=sys.stderr); sys.exit(2)
    merged = collections.defaultdict(lambda: dict(count=0, total_ms=0.0)); out = {}
    for f in files:
        try:
            s = summarize(load(f))
        except Exception as ex:  # a partial/corrupt trace should not hide the others
            print(f"== {f}: unreadable ({ex})"); continue
        out[os.path.basename(f)] = {k: v for k, v in s.items() if k != "table"} | dict(top=sorted(s["table"].items(), key=lambda kv: -kv[1]["total_ms"])[: a.top])
        print_table(os.path.basename(f), s, a.top, a.steps)
        for k, v in s["table"].items():
            merged[k]["count"] += v["count"]; merged[k]["total_ms"] += v["total_ms"]
    if len(files) > 1:
        tot = sum(v["total_ms"] for v in merged.values())
        print_table("ALL RANKS merged", dict(span_ms=0, kernels=sum(v["count"] for v in merged.values()), kernel_time_ms=tot, graph_launches=0, table=merged), a.top, a.steps)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
