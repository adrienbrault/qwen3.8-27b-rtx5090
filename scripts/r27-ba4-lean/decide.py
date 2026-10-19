#!/usr/bin/env python3
"""BA4 lean gate verdict (OPERATIONS §18: numerics-changing kernel = screen + decode check + function smoke).

  decide.py <results dir>     -> prints VERDICT json, exit 0 on PASS, 1 on FAIL

Rule, written before the data (2026-10-19):
  1. Activation: the BA4 engine log shows `SM12X BA4 ACTIVE` for all 48 in_proj_ba prefixes on both TP ranks, `USED` for
     all 48, and no `SM12X BA4 INACTIVE` line.
  2. Decode agreement vs the L1a daily (greedy, r156 corpus, 20 chunks x 512 tokens, ctx 0 and 30,000): every chunk where
     BA4 first diverges from L1a run 1 must diverge at a near-tie: L1a's top-1 minus top-2 logprob at that position
     <= 0.5 nat. L1a run 1 vs run 2 (same boot) is the control and is reported, not gated.
  3. Tool calls: BA4 passes every tool-call smoke request L1a passes.
  4. Completeness: no chunk missing on either side.
"""
import json, re, sys
from pathlib import Path

NEAR_TIE_NAT = 0.5
LAYERS = 48


def load(p):
    return {(j["id"], j["ctx"]): j for j in map(json.loads, open(p))}


def gap(row, i):
    top = (row.get("top") or [None] * (i + 1))[i]
    if not top or len(top) < 2:
        return None
    v = sorted(top.values(), reverse=True)
    return v[0] - v[1]


def compare(ref, other):
    out = {"chunks": 0, "agree": 0, "missing": len(set(ref) ^ set(other)), "divergences": [], "dlp": []}
    for k in sorted(set(ref) & set(other)):
        a, b = ref[k], other[k]
        out["chunks"] += 1
        n = min(len(a["tokens"]), len(b["tokens"]))
        div = next((i for i in range(n) if a["tokens"][i] != b["tokens"][i]), None)
        for i in range(n if div is None else div):
            if a["logprobs"][i] is not None and b["logprobs"][i] is not None:
                out["dlp"].append(abs(a["logprobs"][i] - b["logprobs"][i]))
        if div is None and len(a["tokens"]) != len(b["tokens"]):
            div = n
        if div is None:
            out["agree"] += 1
        else:
            g = gap(a, div) if div < len(a["tokens"]) else None
            out["divergences"].append({"chunk": k[0], "ctx": k[1], "pos": div, "ref_top12_gap": g,
                                       "near_tie": g is not None and g <= NEAR_TIE_NAT})
    d = sorted(out.pop("dlp"))
    out["median_abs_dlogprob_agreed"] = d[len(d) // 2] if d else None
    out["p99_abs_dlogprob_agreed"] = d[int(len(d) * 0.99)] if d else None
    return out


def main(r):
    r = Path(r)
    v = {"rule": __doc__.split("Rule, written before the data (2026-10-19):")[1].strip()}
    log = (r / "BA4-engine.log").read_text(errors="replace")
    active = {}
    for m in re.finditer(r"SM12X BA4 ACTIVE: prefix=(\S+) rank=(\d+)", log):
        active.setdefault(m.group(2), set()).add(m.group(1))
    used = set(re.findall(r"SM12X BA4 USED: prefix=(\S+)", log))
    inactive = re.findall(r"SM12X BA4 INACTIVE: prefix=(\S+) pid=\d+ reason=(\S+)", log)
    v["activation"] = {"active_per_rank": {k: len(s) for k, s in active.items()}, "used": len(used),
                       "inactive": inactive[:10]}
    act_ok = (sorted(active) == ["0", "1"] and all(len(s) == LAYERS for s in active.values())
              and len(used) == LAYERS and not inactive)
    ctl, cand = [], []
    for ctx in (0, 30000):
        a1, a2, b = (load(r / f"{arm}-ctx{ctx}.jsonl") for arm in ("L1a-1", "L1a-2", "BA4"))
        ctl.append({"ctx": ctx, **compare(a1, a2)})
        cand.append({"ctx": ctx, **compare(a1, b)})
    v["control_L1a_vs_L1a"] = ctl
    v["candidate_L1a_vs_BA4"] = cand
    dec_ok = all(c["missing"] == 0 and all(d["near_tie"] for d in c["divergences"]) for c in cand)
    tl, tb = (json.load(open(r / f"{arm}-tools.json")) for arm in ("L1a", "BA4"))
    tool_ok = all(rb["ok"] for ra, rb in zip(tl["rows"], tb["rows"]) if ra["ok"])
    v["tools"] = {"L1a": f'{tl["passed"]}/{tl["total"]}', "BA4": f'{tb["passed"]}/{tb["total"]}'}
    v["checks"] = {"activation": act_ok, "decode_near_tie_only": dec_ok, "tools": tool_ok}
    v["verdict"] = "PASS" if act_ok and dec_ok and tool_ok else "FAIL"
    print(json.dumps(v, indent=1))
    return 0 if v["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
