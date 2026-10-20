#!/usr/bin/env python3
"""r27 flag screen decider (rules in r27-flags-screen.sh's header, written before the data 2026-10-20).

  decide.py <results dir>   -> prints the per-arm table and one VERDICT line per candidate arm
"""
import glob, json, re, statistics as st, sys
from pathlib import Path


def decode(r):
    out = {}   # (arm, kind, c) -> [(tps, acc)] one per boot
    for f in glob.glob(str(r / "decode-*-*.jsonl")):
        _, idx, arm, kind = Path(f).stem.split("-")
        for line in open(f):
            s = json.loads(line)["summary"]
            out.setdefault((arm, kind, s["c"]), []).append((s["ss_per_stream_tps_median"], s["accept_per_draft_median"]))
    return out


def reask(r):
    out = {}   # (arm, ctx) -> warm-send ttfts (sends 2..)
    for f in glob.glob(str(r / "reask-*-*.out")):
        _, idx, arm, ctx = Path(f).stem.split("-")
        for line in open(f):
            if line.startswith("RESULT "):
                sends = json.loads(line[7:])["sends"]
                out.setdefault((arm, int(ctx)), []).extend(s["ttft_s"] for s in sends[1:])
                out.setdefault((arm, int(ctx), "cold"), []).append(sends[0]["ttft_s"])
    return out


def ident(r):
    out = {}   # (arm, ctx) -> [identical, n, warm ttfts, cold ttfts]
    for f in glob.glob(str(r / "ident-*-*.out")):
        _, idx, arm, ctx = Path(f).stem.split("-")
        for line in open(f):
            if line.startswith("RESULT "):
                j = json.loads(line[7:]); o = out.setdefault((arm, int(ctx)), [0, 0, [], []])
                o[0] += j["identical"]; o[1] += j["n"]; o[2] += j["ttft_warm"]; o[3] += j["ttft_cold"]
    return out


def main(r):
    r = Path(r)
    if glob.glob(str(r / "ident-*.out")):
        q = ident(r); ok = True
        for ctx in (6000, 32000):
            c, e = q[("C", ctx)], q[("E", ctx)]
            print(f"ident {ctx:>5}: C {c[0]}/{c[1]} identical, warm TTFT {st.median(c[2]):.3f} s (cold {st.median(c[3]):.2f}); "
                  f"E {e[0]}/{e[1]}, warm {st.median(e[2]):.3f} s (cold {st.median(e[3]):.2f})")
            ok &= e[0] >= c[0] - 1 and st.median(e[2]) < st.median(c[2])
        print("VERDICT E identity:", "PASS" if ok else "FAIL")
        return
    d, q = decode(r), reask(r)
    med = lambda k, i: st.median(v[i] for v in d[k])
    cells = sorted({(k, c) for (_, k, c) in d})
    print("cell            C t/s   P t/s   E t/s |  C acc   P acc   E acc")
    for k, c in cells:
        row = [med((a, k, c), 0) if (a, k, c) in d else float("nan") for a in "CPE"]
        spread = {a: [round(v[0], 1) for v in d[(a, k, c)]] for a in "CPE" if (a, k, c) in d}
        acc = [med((a, k, c), 1) if (a, k, c) in d else float("nan") for a in "CPE"]
        print(f"{k:6} c{c:<3}  " + " ".join(f"{x:7.1f}" for x in row) + " | " + " ".join(f"{x:6.3f}" for x in acc) + f"   boots {spread}")
    for ctx in (6000, 32000) if ("C", 6000) in q else ():
        print(f"reask {ctx:>5}: warm TTFT median C {st.median(q[('C', ctx)]):.3f} s, P {st.median(q[('P', ctx)]):.3f}, "
              f"E {st.median(q[('E', ctx)]):.3f}; cold C {st.median(q[('C', ctx, 'cold')]):.2f}, E {st.median(q[('E', ctx, 'cold')]):.2f}")
    t = lambda a, k, c: med((a, k, c), 0)
    a = lambda a_, k, c: med((a_, k, c), 1)
    if ("P", "code", 1) not in d:
        # Round 4 (E decode no-regression, 10 runs, 3 boots per arm): rule written before the data — every cell within 2 %.
        gains = {f"{k} c{c}": round(100 * (t("E", k, c) / t("C", k, c) - 1), 1) for k, c in cells}
        ok = all(abs(v) <= 2.0 for v in gains.values())
        print("VERDICT E decode:", "PASS" if ok else "FAIL", json.dumps(gains))
        return
    p_ok = (all(a("P", k, 1) - a("C", k, 1) >= 0.015 for k in ("code", "prose"))
            and all(t("P", k, c) >= 0.98 * t("C", k, c) for k, c in cells)
            and all(t("P", k, 1) >= 1.01 * t("C", k, 1) for k in ("code", "prose")))
    gains = {f"{k} c{c}": round(100 * (t("P", k, c) / t("C", k, c) - 1), 1) for k, c in cells}
    print("VERDICT P (probabilistic drafting):", "PASS" if p_ok else "FAIL", json.dumps(gains))
    if ("E", "code", 1) not in d:
        return
    e32 = st.median(q[("C", 32000)]) - st.median(q[("E", 32000)])
    e6 = st.median(q[("C", 6000)]) - st.median(q[("E", 6000)])
    e_ok = e32 >= 0.08 and e6 >= 0 and all(abs(t("E", k, c) / t("C", k, c) - 1) <= 0.02 for k, c in cells)
    print(f"VERDICT E (no eagle block drop): {'PASS' if e_ok else 'FAIL'} warm TTFT -{e32:.3f} s at 32K, -{e6:.3f} s at 6K")


if __name__ == "__main__":
    main(sys.argv[1])
