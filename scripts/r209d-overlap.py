#!/usr/bin/env python3
"""R209d: rebuild each request's prefill/decode phase timeline from the bench client's per-request records and MEASURE the
overlap, replacing R209's counter-based claim (which was structural -- see r209d-overlap.sh for the source trail).

Input: <results-dir>/<row>.json written by `vllm bench serve --save-result --save-detailed`, carrying per-request
start_times, ttfts and itls (all seconds, perf_counter epoch). Request i occupies its PREFILL phase over
[start_i, start_i+ttft_i) and its DECODE phase over [start_i+ttft_i, +sum(itls_i)). Sampling those intervals on a 50 ms grid
gives n_prefill(t) and n_decode(t) with no engine accounting in the path.

Two questions, kept separate:
  1. REGIME -- what fraction of wall time had at least one request in each phase simultaneously.
  2. MECHANISM -- do the long decode stalls land while a prefill is in flight? Reported as the CONDITIONAL stall rate:
     every inter-token gap is classified by n_prefill at its midpoint, and P(stall | prefill in flight) is compared with
     P(stall | none). Comparing a stall INTERVAL against the per-cell share of prefill time instead would overstate the
     effect, since a 0.5-0.9 s window touches a prefill cell by chance far more often than a single cell does.
  3. Is the TTFT tail a start-up artefact? Reported from arrival spread and the p99 request's arrival rank -- simultaneous
     client starts and Poisson arrivals filling the concurrency cap are different shapes and get different sentences.
Usage: r209d-overlap.py <results-dir>
"""
import csv, datetime, json, os, statistics, sys

R = sys.argv[1]
DT = 0.05
CONC = 8  # --max-concurrency used by both rows


def load(path):
    d = json.load(open(path))
    st, tt, itl = d.get("start_times"), d.get("ttfts"), d.get("itls")
    if not st or not tt or not itl:
        return None
    reqs = []
    for s, t, il in zip(st, tt, itl):
        il = [x for x in il if x is not None]
        reqs.append(dict(p0=s, p1=s + t, d1=s + t + sum(il), itls=il, ttft=t))
    return reqs


def grid(reqs, lo, hi):
    n = max(1, int((hi - lo) / DT))
    pre, dec = [0] * n, [0] * n
    for r in reqs:
        for arr, a, b in ((pre, r["p0"], r["p1"]), (dec, r["p1"], r["d1"])):
            i, j = max(0, int((a - lo) / DT)), min(n, int((b - lo) / DT) + 1)
            for k in range(i, j):
                arr[k] += 1
    return pre, dec


def phases(pre, dec):
    n = len(pre)
    both = sum(1 for a, b in zip(pre, dec) if a and b)
    ponly = sum(1 for a, b in zip(pre, dec) if a and not b)
    donly = sum(1 for a, b in zip(pre, dec) if b and not a)
    idle = n - both - ponly - donly
    return dict(n=n, secs=round(n * DT, 1), both=round(both / n, 3), prefill_only=round(ponly / n, 3),
                decode_only=round(donly / n, 3), idle=round(idle / n, 3),
                mean_pre=round(sum(pre) / n, 2), mean_dec=round(sum(dec) / n, 2))


def stalls(reqs, lo, pre):
    # The null must NOT be the per-cell share of time with a prefill in flight: a stall is an INTERVAL of ~0.5-0.9 s, and
    # "did any cell inside it carry a prefill" is satisfied by chance far more often than a single cell is, so comparing
    # against the per-cell rate overstates the effect. Classify each gap by n_prefill at its MIDPOINT instead and report the
    # conditional stall rate with and without a prefill in flight -- interval length drops out of that comparison entirely.
    allitl = [x for r in reqs for x in r["itls"]]
    if not allitl:
        return None
    med = statistics.median(allitl)
    thr = 5 * med
    n = len(pre)
    hit = tot = 0
    for r in reqs:
        t = r["p1"]
        for x in r["itls"]:
            if x > thr:
                tot += 1
                i, j = max(0, int((t - lo) / DT)), min(n, int((t + x - lo) / DT) + 1)
                if any(pre[k] for k in range(i, j)):
                    hit += 1
            t += x
    base = sum(1 for v in pre if v) / n
    big = [x for x in allitl if x > thr]
    with_pre = no_pre = stall_with = stall_no = 0
    for r in reqs:
        t = r["p1"]
        for x in r["itls"]:
            ci = min(n - 1, max(0, int((t + x / 2 - lo) / DT)))
            if pre[ci]:
                with_pre += 1
                stall_with += x > thr
            else:
                no_pre += 1
                stall_no += x > thr
            t += x
    return dict(gaps_with_prefill=with_pre, gaps_without_prefill=no_pre,
                stalls_with_prefill=stall_with, stalls_without_prefill=stall_no,
                p_stall_given_prefill=round(stall_with / with_pre, 4) if with_pre else None,
                p_stall_given_no_prefill=round(stall_no / no_pre, 4) if no_pre else None,median_itl_ms=round(med * 1000, 2), threshold_ms=round(thr * 1000, 1), stalls=tot,
                total_itls=len(allitl), stall_rate=round(tot / len(allitl), 4),
                one_in=round(len(allitl) / tot, 1) if tot else None,
                mean_stall_ms=round(statistics.mean(big) * 1000, 1) if big else None,
                median_stall_ms=round(statistics.median(big) * 1000, 1) if big else None,
                stall_time_s=round(sum(big), 1) if big else None,
                decode_time_s=round(sum(allitl), 1),
                stall_share_of_decode=round(sum(big) / sum(allitl), 3) if big else None,
                coincide_with_prefill=round(hit / tot, 3) if tot else None, base_rate=round(base, 3))


S = []
if os.path.exists(f"{R}/power.csv"):
    for row in csv.reader(open(f"{R}/power.csv")):
        # this unit's sampler emits 6 columns (timestamp,index,power.draw,clocks.sm,temperature.gpu,utilization.gpu).
        # r208-power-summary.py reads row[6] because R208's sampler also logged clocks.mem -- do not copy its indices.
        if len(row) < 6:
            continue
        try:
            S.append((datetime.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp(),
                      int(row[1]), float(row[2]), float(row[3]), float(row[5])))
        except Exception:
            pass

MARKS = {}
if os.path.exists(f"{R}/marks.csv"):
    for m in csv.DictReader(open(f"{R}/marks.csv")):
        MARKS[m["row"]] = (float(m["t_start"]), float(m["t_end"]))

out = {}
for row in ("mix-8k", "mix-32k-deep"):
    p = f"{R}/{row}.json"
    if not os.path.exists(p):
        print(f"{row}: MISSING {p}")
        continue
    reqs = load(p)
    if reqs is None:
        print(f"{row}: JSON has no per-request start_times/ttfts/itls (was --save-detailed passed?)")
        continue
    lo, hi = min(r["p0"] for r in reqs), max(r["d1"] for r in reqs)
    pre, dec = grid(reqs, lo, hi)
    rec = {"full": phases(pre, dec)}
    # STEADY STATE: start where the concurrency slots are actually saturated -- the CONC-th first token, not the first, since
    # at the first token 7 of 8 requests are still in the start-up ramp. Ends at the last request's arrival.
    a = sorted(r["p1"] for r in reqs)[min(CONC, len(reqs)) - 1] - lo
    b = max(r["p0"] for r in reqs) - lo
    if b > a:
        i, j = int(a / DT), int(b / DT)
        rec["steady"] = phases(pre[i:j], dec[i:j])
        rec["steady_window_s"] = [round(a, 1), round(b, 1)]
    rec["stalls"] = stalls(reqs, lo, pre)
    # Is the TTFT tail a START-UP artefact or steady-state queueing? Simultaneous starts (the 8K row) and Poisson arrivals
    # filling the concurrency cap (the 32K row, --request-rate 0.6) are DIFFERENT shapes and must not share a sentence.
    starts = sorted(r["p0"] for r in reqs)
    kk = int(0.99 * (len(reqs) - 1))
    byttft = sorted(range(len(reqs)), key=lambda i: reqs[i]["ttft"])[kk]
    order = sorted(range(len(reqs)), key=lambda i: reqs[i]["p0"])
    rec["arrivals"] = dict(first_conc_spread_s=round(starts[min(CONC, len(reqs)) - 1] - starts[0], 3),
                           p99_ttft_s=round(reqs[byttft]["ttft"], 2),
                           p99_arrived_at_s=round(reqs[byttft]["p0"] - lo, 2),
                           p99_arrival_rank=order.index(byttft) + 1, n=len(reqs))
    rec["requests"] = len(reqs)
    rec["sum_ttft_s"] = round(sum(r["ttft"] for r in reqs), 1)
    rec["window_s"] = round(hi - lo, 1)
    rec["ttft_occupancy"] = round(sum(r["ttft"] for r in reqs) / (hi - lo), 2)
    if row in MARKS:
        a2, b2 = MARKS[row]
        for idx in (0, 1):
            v = [(pw, cl) for (t, i, pw, cl, u) in S if a2 <= t <= b2 and i == idx and u >= 50]
            if v:
                ps = sorted(pw for pw, _ in v)
                rec[f"gpu{idx}_busy_W"] = dict(mean=round(sum(ps) / len(ps), 1),
                                               p95=round(ps[int(0.95 * (len(ps) - 1))], 1), max=round(ps[-1], 1),
                                               sm=round(sum(c for _, c in v) / len(v)))
    out[row] = rec

json.dump(out, open(f"{R}/overlap.json", "w"), indent=1)
print(f"\n== prefill/decode phase timeline (50 ms grid, client-side; n_prefill and n_decode are simultaneous request counts) ==")
hdr = f"{'row':<14}{'window':>9}{'both':>8}{'pre_only':>10}{'dec_only':>10}{'idle':>7}{'mean_pre':>10}{'mean_dec':>10}"
for scope in ("full", "steady"):
    print(f"\n-- {scope} --")
    print(hdr)
    for row, rec in out.items():
        d = rec.get(scope)
        if not d:
            continue
        print(f"{row:<14}{d['secs']:>9.1f}{d['both']:>8.3f}{d['prefill_only']:>10.3f}{d['decode_only']:>10.3f}"
              f"{d['idle']:>7.3f}{d['mean_pre']:>10.2f}{d['mean_dec']:>10.2f}")

print("\n== decode stalls vs in-flight prefill (mechanism) ==")
print(f"{'row':<14}{'med_itl':>9}{'one_in':>8}{'mean_ms':>9}{'p50_ms':>8}{'%dec':>7}{'P(stall|pre)':>22}{'P(stall|no pre)':>22}")
for row, rec in out.items():
    st = rec.get("stalls")
    if not st:
        continue
    print(f"{row:<14}{st['median_itl_ms']:>9.2f}{st['one_in']:>8.1f}{st['mean_stall_ms']:>9.1f}"
          f"{st['median_stall_ms']:>8.1f}{st['stall_share_of_decode']:>7.3f}"
          f"{st['p_stall_given_prefill']:>13.4f} ({st['stalls_with_prefill']}/{st['gaps_with_prefill']})"
          f"{st['p_stall_given_no_prefill']:>13.4f} ({st['stalls_without_prefill']}/{st['gaps_without_prefill']})")

print("\n== TTFT tail: start-up ramp or steady state? ==")
for row, rec in out.items():
    a = rec.get("arrivals")
    if a:
        print(f"{row:<14} first {CONC} arrivals spread {a['first_conc_spread_s']:.3f}s; p99 TTFT {a['p99_ttft_s']:.2f}s held "
              f"by the request arriving at t={a['p99_arrived_at_s']:.2f}s (arrival rank {a['p99_arrival_rank']} of {a['n']})")

print("\n== occupancy (Little's law -- MAGNITUDE ONLY, not evidence of interleaving) ==")
for row, rec in out.items():
    print(f"{row:<14} sum(TTFT)={rec['sum_ttft_s']:>7.1f}s over window {rec['window_s']:>6.1f}s "
          f"-> mean {rec['ttft_occupancy']:.2f} requests in their arrival->first-token phase")

print("\n== power while busy (util >= 50) ==")
for row, rec in out.items():
    for i in (0, 1):
        d = rec.get(f"gpu{i}_busy_W")
        if d:
            print(f"{row:<14} GPU{i}: mean {d['mean']:.0f} W  p95 {d['p95']:.0f} W  max {d['max']:.0f} W  sm {d['sm']} MHz")
