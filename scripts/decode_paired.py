#!/usr/bin/env python3
"""Per-chunk paired read of two decode_fidelity dumps against a reference (R180): for every chunk, the greedy agreed-prefix
length and the median |Δlogprob| on the agreed prefix, arm A vs arm B, and how many chunks each arm is closer on.
usage: decode_paired.py REF A B [--label-a X --label-b Y]"""
import argparse, json, statistics
ap = argparse.ArgumentParser(); ap.add_argument("ref"); ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B"); ap.add_argument("--quiet", action="store_true"); args = ap.parse_args()
def load(p): return {d["id"]: d for d in (json.loads(l) for l in open(p))}
ref, A, B = load(args.ref), load(args.a), load(args.b)
def stat(r, a):
    n = 0
    while n < min(len(r["tokens"]), len(a["tokens"])) and r["tokens"][n] == a["tokens"][n]: n += 1
    d = [abs(r["logprobs"][i] - a["logprobs"][i]) for i in range(n)]
    return n, (statistics.median(d) if d else None), d
closer = {args.label_a: 0, args.label_b: 0, "tie": 0}; alld = {args.label_a: [], args.label_b: []}; rows = []
for i in sorted(ref):
    if i not in A or i not in B: continue
    na, ma, da = stat(ref[i], A[i]); nb, mb, db = stat(ref[i], B[i])
    alld[args.label_a] += da; alld[args.label_b] += db
    if ma is None or mb is None: continue
    w = "tie" if ma == mb else (args.label_a if ma < mb else args.label_b); closer[w] += 1
    rows.append(f"  {i:3d} {ref[i].get('bucket','?'):8s} n={na:3d}/{nb:3d}  {ma:.5f} / {mb:.5f}  closer: {w}")
if not args.quiet: print("\n".join(rows))
print(json.dumps({"chunks": len(rows), "closer": closer,
    "pooled_median_abs_dlogprob": {k: (round(statistics.median(v), 5) if v else None) for k, v in alld.items()},
    "pooled_p90": {k: (round(sorted(v)[int(0.9 * len(v))], 5) if v else None) for k, v in alld.items()},
    "agreed_tokens": {k: len(v) for k, v in alld.items()}}))
