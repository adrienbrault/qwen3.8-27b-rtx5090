#!/usr/bin/env python3
"""BA4 kernel-error decider: is BA4's product at least as close to the exact one as the original bf16 GEMM's?

  kernel_decide.py <errstat dir>   -> prints VERDICT json, exit 0 on PASS, 1 on FAIL

Rule, written before the data (2026-10-20), on live eager M=4 decode-verify operands (errstat_patch.py), all ranks:
  PASS iff  aggregate mean |err(BA4)| <= 1.05 x aggregate mean |err(original)|  and the same for rms error,
       and  no layer has mean |err(BA4)| > 1.5 x its mean |err(original)|,
       and  BA4 is unbiased: |mean signed err(BA4)| <= max(0.1 x mean |err(BA4)|, |mean signed err(original)| x 1.5)
       and  at least 48 layers x 2 ranks seen with >= 50 calls each.
best_bf16 (exact rounded once) is reported as the floor no bf16 GEMM can beat.
"""
import glob, json, sys

KEYS = ("ba4", "original", "best_bf16")


def main(d):
    per = {}
    for f in glob.glob(f"{d}/rank*.json"):
        rank = f.split("/")[-1].split("-")[0]
        for prefix, a in json.load(open(f))["layers"].items():
            per[(rank, prefix)] = a
    def agg(rows, k, i):
        n = sum(a["n"] for a in rows)
        return sum(a[k][i] for a in rows) / n
    rows = list(per.values())
    calls_ok = len(per) >= 96 and all(a["n"] >= 50 * 4 * 48 for a in rows)
    v = {"rule": __doc__.split("Rule, written before the data (2026-10-20), ")[1].strip(),
         "rank_layers": len(per), "min_elements": min((a["n"] for a in rows), default=0)}
    if not rows:
        v["verdict"] = "FAIL"; print(json.dumps(v)); return 1
    for k in KEYS:
        v[k] = {"mean_abs": agg(rows, k, 0), "rms": agg(rows, k, 1) ** 0.5, "bias": agg(rows, k, 2),
                "max_abs": max(a[k][3] for a in rows)}
    v["mean_abs_output"] = sum(a["scale"] for a in rows) / sum(a["n"] for a in rows)
    worst = max(((a["ba4"][0] / max(a["original"][0], 1e-30)), p) for (_, p), a in per.items())
    v["worst_layer_ratio"] = {"ratio": worst[0], "layer": worst[1]}
    b, o = v["ba4"], v["original"]
    v["ratio_mean_abs"] = b["mean_abs"] / o["mean_abs"]
    v["ratio_rms"] = b["rms"] / o["rms"]
    checks = {"mean_abs": v["ratio_mean_abs"] <= 1.05, "rms": v["ratio_rms"] <= 1.05, "worst_layer": worst[0] <= 1.5,
              "unbiased": abs(b["bias"]) <= max(0.1 * b["mean_abs"], 1.5 * abs(o["bias"])), "coverage": calls_ok}
    v["checks"] = checks
    v["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    print(json.dumps(v, indent=1))
    return 0 if v["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
