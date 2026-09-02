#!/usr/bin/env python3
"""CPU-side view of a torch-profiler trace: where does GPU idle time come from?

  python3 prof_cpu.py trace.json.gz [--steps 60] [--gap-us 150] [--top 30]

Merges all GPU kernel intervals, finds idle gaps longer than --gap-us, and attributes each gap to
the CPU ops (cat cpu_op / user_annotation / cuda_runtime) running during it. Also prints the CUDA
runtime API histogram (launches, memcpys, syncs) and the top CPU ops by total duration, per step.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re


def short(n, w=90):
    n = re.sub(r"\(.*", "", n).replace("void ", "")
    return n if len(n) <= w else n[: w - 3] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--steps", type=float, default=60)
    ap.add_argument("--gap-us", type=float, default=150)
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()
    opener = gzip.open if a.trace.endswith(".gz") else open
    with opener(a.trace, "rt") as f:
        d = json.load(f)
    ev = d["traceEvents"]
    kern = sorted(((e["ts"], e["ts"] + e["dur"]) for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")), key=lambda x: x[0])
    cpu = [e for e in ev if e.get("cat") in ("cpu_op", "user_annotation", "cuda_runtime") and "dur" in e]
    rt = collections.defaultdict(lambda: [0.0, 0])
    ops = collections.defaultdict(lambda: [0.0, 0])
    for e in cpu:
        tgt = rt if e["cat"] == "cuda_runtime" else ops
        tgt[e["name"]][0] += e["dur"]
        tgt[e["name"]][1] += 1
    # merge kernel intervals -> idle gaps
    gaps = []
    cur_end = kern[0][1] if kern else 0
    for s, e in kern[1:]:
        if s > cur_end + a.gap_us:
            gaps.append((cur_end, s))
        cur_end = max(cur_end, e)
    span = (kern[-1][1] - kern[0][0]) if kern else 1
    idle = sum(e - s for s, e in gaps)
    print(f"== {a.trace}")
    print(f"span={span/1000:.1f} ms idle(gaps>{a.gap_us:g}us)={idle/1000:.1f} ms ({100*idle/span:.1f}%)  gaps={len(gaps)}  per step: idle {idle/1000/a.steps:.2f} ms, gaps {len(gaps)/a.steps:.1f}")
    # attribute gaps: for each gap, cpu ops (non-runtime, innermost = shortest overlapping) covering the gap midpoint
    cpu_sorted = sorted((e for e in cpu if e["cat"] != "cuda_runtime"), key=lambda e: e["ts"])
    attr = collections.defaultdict(float)
    hist = collections.Counter()
    import bisect
    starts = [e["ts"] for e in cpu_sorted]
    for s, e in gaps:
        mid = (s + e) / 2
        i = bisect.bisect_right(starts, mid)
        best = None
        for j in range(max(0, i - 400), i):
            c = cpu_sorted[j]
            if c["ts"] <= mid <= c["ts"] + c["dur"]:
                if best is None or c["dur"] < best["dur"]:
                    best = c
        name = short(best["name"]) if best else "<no cpu op>"
        attr[name] += e - s
        hist[name] += 1
    print(f"idle attribution (innermost CPU op at gap midpoint), ms/step:")
    for n, t in sorted(attr.items(), key=lambda kv: -kv[1])[: a.top]:
        print(f"  {t/1000/a.steps:>8.3f}  n={hist[n]/a.steps:>6.1f}  {n}")
    print("cuda runtime API (count/step, ms/step):")
    for n, (t, c) in sorted(rt.items(), key=lambda kv: -kv[1][0])[:15]:
        print(f"  {c/a.steps:>8.1f} {t/1000/a.steps:>8.3f}  {n}")
    print(f"top CPU ops by total duration (ms/step, count/step):")
    for n, (t, c) in sorted(ops.items(), key=lambda kv: -kv[1][0])[: a.top]:
        print(f"  {t/1000/a.steps:>8.3f} {c/a.steps:>8.1f}  {short(n)}")


if __name__ == "__main__":
    main()
