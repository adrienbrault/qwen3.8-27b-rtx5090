#!/usr/bin/env python3
"""BA4 tool-eval A/B decider (user 2026-10-20: "maybe compare with tool eval ?!").

  tool_decide.py <results dir>   -> prints VERDICT json, exit 0 on PASS, 1 on FAIL

Rule, written before the data (2026-10-20), on tool-eval-bench as the r27 QUALIFY runs it (69 scenarios, sampled
T 0.6 / top-p 0.95 / top-k 20, 4 trials, parallel 8), L1a vs BA4 booted as served (thinking on):
  PASS iff  mean(BA4) >= mean(L1a) - 2.0 points
       and  the two 95 % confidence intervals overlap.
"""
import json, sys
from pathlib import Path


def stats(p):
    t = json.load(open(p))["trial_statistics"]
    return {"mean": t["final_score_mean"], "stddev": t["final_score_stddev"], "ci95": t.get("final_score_ci95")}


def main(r):
    r = Path(r)
    a, b = stats(r / "tooleval-L1a.json"), stats(r / "tooleval-BA4.json")
    overlap = a["ci95"] and b["ci95"] and b["ci95"][1] >= a["ci95"][0] and a["ci95"][1] >= b["ci95"][0]
    v = {"rule": __doc__.split("Rule, written before the data (2026-10-20), ")[1].strip(), "L1a": a, "BA4": b,
         "ci_overlap": bool(overlap)}
    v["verdict"] = "PASS" if b["mean"] >= a["mean"] - 2.0 and overlap else "FAIL"
    print(json.dumps(v, indent=1))
    return 0 if v["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
