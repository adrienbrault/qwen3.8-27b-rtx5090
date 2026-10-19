#!/usr/bin/env python3
"""BA4 task-accuracy decider (the R107d precedent: XQA was cleared by a T=0 GSM8K A/B after its decode deltas).

  gsm_decide.py <results dir>   -> prints VERDICT json, exit 0 on PASS, 1 on FAIL

Rule, written before the data (2026-10-20), on lm-eval `gsm8k` 5-shot, chat template, thinking off, T=0, n=200 (the R299
think-off recipe), flexible-extract exact_match per problem, L1a vs BA4 on the same problems:
  PASS iff  accuracy(BA4) >= accuracy(L1a) - 0.02
       and  the problems only L1a solves are not significantly more than the ones only BA4 solves
            (one-sided exact binomial on the discordant pairs, p >= 0.05).
"""
import glob, json, math, sys
from pathlib import Path

FILTER = "flexible-extract"


def per_problem(d):
    files = glob.glob(str(d / "**" / "samples_gsm8k*.jsonl"), recursive=True)
    if len(files) != 1:
        raise SystemExit(f"{d}: expected one samples file, found {files}")
    out = {}
    for row in map(json.loads, open(files[0])):
        if row.get("filter") == FILTER:
            out[row["doc_id"]] = float(row["exact_match"]) >= 1.0
    return out


def binom_tail(k, n):  # P(X >= k), X ~ Bin(n, 0.5)
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0


def main(r):
    r = Path(r)
    a, b = per_problem(r / "gsm-L1a"), per_problem(r / "gsm-BA4")
    common = sorted(set(a) & set(b))
    if len(common) < 190:
        raise SystemExit(f"only {len(common)} common problems")
    acc_a = sum(a[i] for i in common) / len(common)
    acc_b = sum(b[i] for i in common) / len(common)
    only_a = sum(a[i] and not b[i] for i in common)
    only_b = sum(b[i] and not a[i] for i in common)
    p = binom_tail(only_a, only_a + only_b)
    v = {"rule": __doc__.split("Rule, written before the data (2026-10-20), ")[1].strip(),
         "problems": len(common), "acc_L1a": round(acc_a, 4), "acc_BA4": round(acc_b, 4),
         "only_L1a_correct": only_a, "only_BA4_correct": only_b, "p_one_sided": round(p, 4)}
    v["verdict"] = "PASS" if acc_b >= acc_a - 0.02 and p >= 0.05 else "FAIL"
    print(json.dumps(v, indent=1))
    return 0 if v["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
