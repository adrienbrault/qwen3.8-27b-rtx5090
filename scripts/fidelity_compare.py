#!/usr/bin/env python3
"""R156 fidelity comparator — bucketed top-1 flip rates + PPL delta vs a reference arm.

Consumes the arm-agnostic JSONL dumps written by fidelity_ladder.py. One record per
scored (teacher-forced) position:

    {"doc": int, "pos": int, "tok": str, "tok_lp": float,
     "argmax": str, "p1": float, "top": [[tok, logprob], ...]}

Reference buckets are defined by the REFERENCE arm's p1 (never the arm under test):

    certain    p1 > 0.99
    confident  0.90 < p1 <= 0.99
    moderate   0.50 < p1 <= 0.90
    near-tie   p1 <= 0.50

Primary metric per bucket: top-1 flip rate = fraction of positions where the arm's
argmax != the reference's argmax. Also reports exact corpus PPL per arm (from the
forced-token logprobs, no truncation), a truncated KL over the reference's top-k
support, and depth slices (compounding shows up as the gap widening with position).

Usage:
  fidelity_compare.py --ref dump-A.jsonl --arm dump-C.jsonl [--label "A->C"]
                      [--depth-bins 0,1000,4000,16000,65536] [--json out.json]

A "noise floor" run is just this with two dumps of the SAME arm from separate boots.
"""
import argparse, json, math, sys
from collections import defaultdict

BUCKETS = [
    ("certain",   lambda p: p > 0.99),
    ("confident", lambda p: 0.90 < p <= 0.99),
    ("moderate",  lambda p: 0.50 < p <= 0.90),
    ("near-tie",  lambda p: p <= 0.50),
]


def load(path):
    """(doc,pos) -> record. Skips malformed/None-logprob positions (e.g. first token)."""
    out, bad = {}, 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if r.get("argmax") is None or r.get("p1") is None:
                bad += 1
                continue
            out[(r["doc"], r["pos"])] = r
    return out, bad


def wilson(k, n, z=1.96):
    """Wilson score interval — honest CIs at the 0.1% rates we care about."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def trunc_kl(ref_top, arm_top):
    """KL(ref||arm) over the reference's top-k support only. Truncated: a lower bound,
    comparable across arms scored with identical k, not an absolute divergence."""
    if not ref_top or not arm_top:
        return None
    arm = {t: lp for t, lp in arm_top}
    total = 0.0
    for t, lp in ref_top:
        if t not in arm:
            return None  # support miss: undefined rather than silently wrong
        p = math.exp(lp)
        total += p * (lp - arm[t])
    return total


def ppl(records):
    """Exact corpus perplexity from forced-token logprobs (full vocab, untruncated)."""
    lps = [r["tok_lp"] for r in records.values() if r.get("tok_lp") is not None]
    if not lps:
        return None, 0
    return math.exp(-sum(lps) / len(lps)), len(lps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference arm dump (e.g. bf16 = arm A)")
    ap.add_argument("--arm", required=True, help="arm under test dump")
    ap.add_argument("--label", default=None)
    ap.add_argument("--depth-bins", default="0,1000,4000,16000,65536,262144")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    ref, ref_bad = load(a.ref)
    arm, arm_bad = load(a.arm)
    keys = sorted(set(ref) & set(arm))
    if not keys:
        print("FATAL: no overlapping (doc,pos) positions — arms did not score the same corpus", file=sys.stderr)
        return 2

    label = a.label or f"{a.ref} -> {a.arm}"
    bins = [int(x) for x in a.depth_bins.split(",")]

    # ---- tokenization-alignment guard -------------------------------------
    # The join is on (doc,pos). If the two arms tokenized the corpus differently,
    # we would silently compare unrelated positions and produce a plausible but
    # meaningless number. Verify the forced token matches at every position.
    misaligned = [k for k in keys if ref[k].get("tok") != arm[k].get("tok")]
    if misaligned:
        frac = len(misaligned) / len(keys)
        print(f"FATAL: tokenization misalignment at {len(misaligned):,}/{len(keys):,} "
              f"positions ({100*frac:.3f}%) — the arms did not tokenize the corpus "
              f"identically, so per-position comparison is invalid.", file=sys.stderr)
        for k in misaligned[:5]:
            print(f"   doc={k[0]} pos={k[1]}  ref={ref[k].get('tok')!r}  arm={arm[k].get('tok')!r}",
                  file=sys.stderr)
        return 2

    per_bucket = defaultdict(lambda: [0, 0])   # name -> [flips, n]
    per_depth = defaultdict(lambda: [0, 0])
    kls, kl_undef = [], 0
    flips_total = 0

    for k in keys:
        r, m = ref[k], arm[k]
        flip = 1 if r["argmax"] != m["argmax"] else 0
        flips_total += flip
        p1 = r["p1"]
        for name, pred in BUCKETS:
            if pred(p1):
                per_bucket[name][0] += flip
                per_bucket[name][1] += 1
                break
        pos = k[1]
        lo = max([b for b in bins if b <= pos], default=bins[0])
        per_depth[lo][0] += flip
        per_depth[lo][1] += 1
        kl = trunc_kl(r.get("top"), m.get("top"))
        if kl is None:
            kl_undef += 1
        else:
            kls.append(kl)

    n = len(keys)
    ref_ppl, ref_n = ppl({k: ref[k] for k in keys})
    arm_ppl, _ = ppl({k: arm[k] for k in keys})

    print(f"=== R156 fidelity compare: {label}")
    print(f"scored positions: {n:,}  (ref-only skipped {ref_bad:,}, arm-only skipped {arm_bad:,})")
    print(f"overall top-1 agreement: {100.0*(n-flips_total)/n:.3f}%   (flips {flips_total:,})")
    if ref_ppl and arm_ppl:
        print(f"corpus PPL  ref {ref_ppl:.4f}  arm {arm_ppl:.4f}   delta {100.0*(arm_ppl-ref_ppl)/ref_ppl:+.3f}%   (n={ref_n:,})")
    if kls:
        kls.sort()
        print(f"truncated KL(ref||arm) over ref top-k: mean {sum(kls)/len(kls):.6f}  median {kls[len(kls)//2]:.6f}"
              f"  (undefined at {kl_undef:,} positions — support miss)")

    print("\nflip rate by REFERENCE confidence bucket (the meaningful axis):")
    print(f"  {'bucket':<10} {'n':>10} {'flips':>8} {'rate':>9}   95% CI")
    rows = {}
    for name, _ in BUCKETS:
        f_, n_ = per_bucket[name]
        if n_ == 0:
            continue
        lo, hi = wilson(f_, n_)
        rows[name] = {"n": n_, "flips": f_, "rate": f_ / n_, "ci": [lo, hi]}
        print(f"  {name:<10} {n_:>10,} {f_:>8,} {100.0*f_/n_:>8.3f}%   [{100*lo:.3f}%, {100*hi:.3f}%]")

    print("\nflip rate by position depth (compounding = rate rising with depth):")
    print(f"  {'from-pos':>10} {'n':>10} {'flips':>8} {'rate':>9}")
    depth_rows = {}
    for lo_ in sorted(per_depth):
        f_, n_ = per_depth[lo_]
        if n_ == 0:
            continue
        depth_rows[lo_] = {"n": n_, "flips": f_, "rate": f_ / n_}
        print(f"  {lo_:>10,} {n_:>10,} {f_:>8,} {100.0*f_/n_:>8.3f}%")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({
                "label": label, "ref": a.ref, "arm": a.arm, "positions": n,
                "top1_agreement": (n - flips_total) / n,
                "ppl_ref": ref_ppl, "ppl_arm": arm_ppl,
                "kl_trunc_mean": (sum(kls) / len(kls)) if kls else None,
                "kl_undefined": kl_undef,
                "buckets": rows, "depth": {str(k): v for k, v in depth_rows.items()},
            }, f, indent=2)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
