#!/usr/bin/env python3
"""R211: per-row deltas of the engine's own histograms, joined across arms.

Each scored row is bracketed by a `metrics-<arm>-<row>-{pre,post}.prom` scrape, so subtracting the two gives that row's
counters in isolation — no client-side reconstruction, and no contamination from the plant row or from another arm.

THE PRIMARY SCORE is the inter-token-latency bucket delta. R209d's stall criterion is an inter-token gap above 5x the row's
median; the median lands in the 0.025-0.05 s bucket in every regime measured so far, so >0.2 s is that same threshold on the
histogram's grid. Stall RATE is the share of gaps above it; stall LENGTH is read off the bucket masses (median by linear
interpolation inside the crossing bucket, mean from bucket midpoints — both approximations bounded by the bucket widths, and
both stated as such rather than quoted to a precision the grid cannot carry).

TTFT is split into queue vs prefill compute, because arms A-D and arm E answer different questions: chunk size moves the
prefill term, an admission cap moves the queue term. Usage: r211-summary.py <results-dir>
"""
import os, re, sys

R = sys.argv[1] if len(sys.argv) > 1 else "."
EDGES = ["0.01", "0.025", "0.05", "0.075", "0.1", "0.15", "0.2", "0.3", "0.4", "0.5", "0.75", "1.0", "2.5", "5.0",
         "7.5", "10.0", "20.0", "40.0", "80.0", "+Inf"]
STALL_EDGE = "0.2"


def parse(path):
    buckets, scalars = {}, {}
    if not os.path.exists(path):
        return None
    for line in open(path, errors="ignore"):
        m = re.match(r'^vllm:inter_token_latency_seconds_bucket\{.*?le="([^"]+)".*?\}\s+([0-9.eE+-]+)', line)
        if m:
            buckets[m.group(1)] = float(m.group(2))
            continue
        m = re.match(r'^vllm:([a-z_]+)(_sum|_count|_total)\{[^}]*\}\s+([0-9.eE+-]+)', line)
        if m:
            scalars[m.group(1) + m.group(2)] = scalars.get(m.group(1) + m.group(2), 0.0) + float(m.group(3))
    return buckets, scalars


def delta(a, b):
    return {k: b.get(k, 0.0) - a.get(k, 0.0) for k in set(a) | set(b)}


def stalls(bd):
    """bd: cumulative-bucket DELTAS. Returns (n_gaps, stall_n, stall_share, median_s, mean_s)."""
    tot = bd.get("+Inf", 0.0)
    if tot <= 0:
        return None
    above = tot - bd.get(STALL_EDGE, 0.0)
    if above <= 0:
        return dict(n=tot, stall_n=0, share=0.0, median=None, mean=None)
    idx = EDGES.index(STALL_EDGE)
    lo, mass, mids = float(STALL_EDGE), [], []
    for e in EDGES[idx + 1:]:
        hi = 100.0 if e == "+Inf" else float(e)
        m = bd.get(e, 0.0) - bd.get(EDGES[EDGES.index(e) - 1], 0.0)
        mass.append((lo, hi, max(m, 0.0)))
        mids.append(((lo + hi) / 2, max(m, 0.0)))
        lo = hi
    half, run, med = above / 2, 0.0, None
    for lo_, hi_, m in mass:
        if run + m >= half and m > 0:
            med = lo_ + (half - run) / m * (hi_ - lo_)
            break
        run += m
    wsum = sum(mid * m for mid, m in mids)
    return dict(n=tot, stall_n=above, share=above / tot, median=med, mean=(wsum / above if above else None))


rows = []
for f in sorted(os.listdir(R)):
    m = re.match(r"metrics-(.+?)-(.+?)-pre\.prom$", f)
    if not m:
        continue
    arm, rw = m.group(1), m.group(2)
    pre, post = parse(f"{R}/{f}"), parse(f"{R}/metrics-{arm}-{rw}-post.prom")
    if not pre or not post:
        continue
    bd, sd = delta(pre[0], post[0]), delta(pre[1], post[1])
    st = stalls(bd)

    def per(num, den):
        d = sd.get(den, 0.0)
        return sd.get(num, 0.0) / d if d > 0 else None
    rows.append(dict(
        arm=arm, row=rw, stall=st,
        ttft=per("time_to_first_token_seconds_sum", "time_to_first_token_seconds_count"),
        queue=per("request_queue_time_seconds_sum", "request_queue_time_seconds_count"),
        prefill=per("request_prefill_time_seconds_sum", "request_prefill_time_seconds_count"),
        decode=per("request_decode_time_seconds_sum", "request_decode_time_seconds_count"),
        itl_mean=per("inter_token_latency_seconds_sum", "inter_token_latency_seconds_count"),
        reqs=sd.get("time_to_first_token_seconds_count", 0.0),
        ptoks=sd.get("prompt_tokens_total", 0.0), gtoks=sd.get("generation_tokens_total", 0.0),
        pfx_q=sd.get("prefix_cache_queries_total", 0.0), pfx_h=sd.get("prefix_cache_hits_total", 0.0)))

# Wall-clock per row from marks.csv
wall = {}
mp = f"{R}/marks.csv"
if os.path.exists(mp):
    for line in open(mp).read().splitlines()[1:]:
        p = line.split(",")
        if len(p) >= 4:
            try:
                wall[(p[0], p[1])] = float(p[3]) - float(p[2])
            except ValueError:
                pass

ARMS = {"A": "MNBT 8192 (5 blk)", "B": "MNBT 4096 (2 blk)", "C": "MNBT 1536 (1 blk)",
        "D": "MNBT 16384 (11 blk)", "E": "MNBT 4096 SEQS32", "A2": "MNBT 8192 repeat"}
for rw in ("steady", "ramp"):
    sel = [r for r in rows if r["row"] == rw]
    if not sel:
        continue
    print(f"\n=== row {rw} ===")
    print(f"{'arm':<4}{'config':<20}{'wall s':>8}{'gaps':>8}{'stall%':>8}{'stall med':>11}{'stall mean':>11}"
          f"{'ITL mean':>10}{'TTFT':>8}{'queue':>8}{'prefill':>9}{'pfx hit':>9}")
    for r in sorted(sel, key=lambda r: list(ARMS).index(r["arm"]) if r["arm"] in ARMS else 99):
        s = r["stall"] or {}
        w = wall.get((r["arm"], rw))
        pfx = (r["pfx_h"] / r["pfx_q"] * 100) if r["pfx_q"] else None
        def f(v, n=3, suf=""):
            return f"{v:.{n}f}{suf}" if v is not None else "-"
        print(f"{r['arm']:<4}{ARMS.get(r['arm'], ''):<20}{f(w,1):>8}{s.get('n',0):>8.0f}"
              f"{(s.get('share') or 0)*100:>7.2f}%{f(s.get('median')):>11}{f(s.get('mean')):>11}"
              f"{f(r['itl_mean']):>10}{f(r['ttft'],2):>8}{f(r['queue'],2):>8}{f(r['prefill'],2):>9}"
              f"{f(pfx,1,'%'):>9}")

    base = next((r for r in sel if r["arm"] == "A"), None)
    if base and base["stall"] and base["stall"].get("median"):
        print(f"\n  chunk-scaling test (stall median relative to arm A = {base['stall']['median']:.3f} s;"
              f" predicted ratios C:B:A:D = 0.2 : 0.4 : 1.0 : 2.2 if the stall IS the block-aligned chunk)")
        for r in sorted(sel, key=lambda r: list(ARMS).index(r["arm"]) if r["arm"] in ARMS else 99):
            s = r["stall"] or {}
            if s.get("median"):
                print(f"    {r['arm']:<3} {ARMS.get(r['arm'],''):<20} ratio {s['median']/base['stall']['median']:.2f}")
