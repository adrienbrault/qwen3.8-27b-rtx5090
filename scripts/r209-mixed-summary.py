#!/usr/bin/env python3
"""R209: join the power sampler, the engine-state sampler and the vllm-bench table per row.

Three things this reports that a bench table alone cannot:
  * `busy` draw (utilization.gpu >= 50) vs `win` draw (whole window). Averaging a whole row window dilutes the loaded reading
    with idle gaps; `busy` is the number that says whether a power cap would bind.
  * `mixed` = the fraction of 1 Hz engine samples in which prompt_tokens_total AND generation_tokens_total BOTH advanced. This
    is the evidence that prefill and decode overlapped. A row with a low `mixed` fraction did not measure the mixed regime and
    its deltas mean nothing for the question asked.
  * arm-vs-arm deltas against A0, with A0b as the control: a cost smaller than the A0b swing is inside the noise floor.

Metrics come from the bench's stdout table, NOT --result-filename: native-l2 is root-owned, so the container could not create
/l2/_r209 and --save-result raised FileNotFoundError after printing a complete table. The table also carries the
speculative-decoding block, which the deltas need to show the arms drafted comparably.
Usage: r209-mixed-summary.py <results-dir>
"""
import csv, datetime, json, re, sys

R = sys.argv[1]

S = []
with open(f"{R}/power.csv") as f:
    for row in csv.reader(f):
        if len(row) < 6:
            continue
        try:
            t = datetime.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
            S.append((t, int(row[1]), float(row[2]), float(row[3]), float(row[5])))
        except Exception:
            pass

E = []
try:
    with open(f"{R}/engine.csv") as f:
        for r in csv.DictReader(f):
            try:
                E.append((float(r["t"]), float(r["running"]), float(r["waiting"]),
                          float(r["prompt_tokens_total"]), float(r["generation_tokens_total"])))
            except Exception:
                pass
except FileNotFoundError:
    pass
E.sort()


def power(a, b, idx, busy):
    v = [(p, c) for (t, i, p, c, u) in S if a <= t <= b and i == idx and (u >= 50 or not busy)]
    if not v:
        return None
    ps = sorted(p for p, _ in v)
    return dict(n=len(ps), mean=round(sum(ps) / len(ps), 1), p95=round(ps[int(0.95 * (len(ps) - 1))], 1),
                sm=round(sum(c for _, c in v) / len(v)))


def mixed(a, b):
    """fraction of consecutive engine samples inside [a,b] where prefill and decode both advanced"""
    w = [e for e in E if a <= e[0] <= b]
    if len(w) < 3:
        return None
    both = pre = dec = 0
    for x, y in zip(w, w[1:]):
        dp, dg = y[3] - x[3], y[4] - x[4]
        if dp > 0 and dg > 0:
            both += 1
        elif dp > 0:
            pre += 1
        elif dg > 0:
            dec += 1
    n = len(w) - 1
    return dict(n=n, both=round(both / n, 2), prefill_only=round(pre / n, 2), decode_only=round(dec / n, 2),
                running_mean=round(sum(e[1] for e in w) / len(w), 1),
                running_max=round(max(e[1] for e in w), 1),
                waiting_mean=round(sum(e[2] for e in w) / len(w), 1))


LABELS = {
    "Successful requests": "completed", "Failed requests": "failed",
    "Benchmark duration (s)": "duration", "Total input tokens": "total_input_tokens",
    "Total generated tokens": "total_output_tokens", "Request throughput (req/s)": "request_throughput",
    "Output token throughput (tok/s)": "output_throughput", "Total token throughput (tok/s)": "total_token_throughput",
    "Mean TTFT (ms)": "mean_ttft_ms", "Median TTFT (ms)": "median_ttft_ms", "P99 TTFT (ms)": "p99_ttft_ms",
    "Mean TPOT (ms)": "mean_tpot_ms", "Median TPOT (ms)": "median_tpot_ms", "P99 TPOT (ms)": "p99_tpot_ms",
    "Median ITL (ms)": "median_itl_ms", "P99 ITL (ms)": "p99_itl_ms",
    "Median E2EL (ms)": "median_e2el_ms", "P99 E2EL (ms)": "p99_e2el_ms",
    "Acceptance rate (%)": "accept_rate", "Acceptance length": "accept_len",
}


def parse_table(path):
    try:
        txt = open(path).read()
    except FileNotFoundError:
        return {"error": "no output file"}
    out = {}
    for lab, key in LABELS.items():
        m = re.search(re.escape(lab) + r"\s*:\s*([0-9.]+)", txt)
        if m:
            out[key] = float(m.group(1))
    return out or {"error": "no table in output"}


rows = []
with open(f"{R}/marks.csv") as f:
    for m in csv.DictReader(f):
        a, b = float(m["t_start"]), float(m["t_end"])
        rec = {"arm": m["arm"], "row": m["row"], "dur_s": round(b - a, 1), "mixed": mixed(a, b)}
        for i in (0, 1):
            rec[f"gpu{i}_win"] = power(a, b, i, False)
            rec[f"gpu{i}_busy"] = power(a, b, i, True)
        rec["bench"] = parse_table(f"{R}/{m['arm']}-{m['row']}.txt")
        rows.append(rec)

print(json.dumps(rows, indent=1))

print("\n=== mixed-regime evidence (both counters advancing per 1 s sample) ===")
print(f"{'arm/row':<18} {'n':>4} {'both':>6} {'pf-only':>8} {'dec-only':>9} {'running':>8} {'waiting':>8}")
for r in rows:
    x = r["mixed"]
    if not x:
        print(f"{r['arm'] + '/' + r['row']:<18} (no engine samples)")
        continue
    print(f"{r['arm'] + '/' + r['row']:<18} {x['n']:>4} {x['both']:>6} {x['prefill_only']:>8} {x['decode_only']:>9} "
          f"{x['running_mean']:>8} {x['waiting_mean']:>8}")

print("\n=== power, busy-only (util>=50) mean / p95 W per card ===")
print(f"{'arm/row':<18} {'gpu0 mean':>10} {'p95':>7} {'gpu1 mean':>10} {'p95':>7} {'sm0':>6}")
for r in rows:
    g0, g1 = r.get("gpu0_busy"), r.get("gpu1_busy")
    if not g0 or not g1:
        continue
    print(f"{r['arm'] + '/' + r['row']:<18} {g0['mean']:>10} {g0['p95']:>7} {g1['mean']:>10} {g1['p95']:>7} {g0['sm']:>6}")

print("\n=== deltas vs A0 (A0b is the control: a cost inside the A0b swing is noise) ===")
base = {r["row"]: r for r in rows if r["arm"] == "A0"}
for metric, label, up in (("output_throughput", "output tok/s", True),
                          ("total_token_throughput", "total tok/s", True),
                          ("median_ttft_ms", "TTFT p50 ms", False),
                          ("p99_ttft_ms", "TTFT p99 ms", False),
                          ("median_tpot_ms", "TPOT p50 ms", False),
                          ("p99_itl_ms", "ITL p99 ms", False),
                          ("accept_len", "MTP accept length", True)):
    print(f"\n-- {label} ({'higher' if up else 'lower'} is better)")
    for r in rows:
        v0 = base.get(r["row"], {}).get("bench", {}).get(metric)
        v = r.get("bench", {}).get(metric)
        if v is None or not v0:
            continue
        print(f"  {r['arm'] + '/' + r['row']:<18} {v:>12.2f}  {(v - v0) / v0 * 100.0:+6.1f}%")
