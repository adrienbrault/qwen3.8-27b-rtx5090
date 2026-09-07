#!/usr/bin/env python3
"""R208: join the continuous nvidia-smi sampler to the per-row epoch marks and report draw per card per row.

The script's inline summary averages over the whole row window, which includes the idle gaps between decode_ss runs and so
UNDERSTATES the loaded draw. This reports both: `win` = the full row window, `busy` = only samples with utilization.gpu >= 50,
which is the number that says whether a cap would bind. Usage: r208-power-summary.py <results-dir>
"""
import csv, datetime, json, os, re, sys

R = sys.argv[1]
S = []
with open(f"{R}/power.csv") as f:
    for row in csv.reader(f):
        if len(row) < 7:
            continue
        try:
            t = datetime.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
            S.append((t, int(row[1]), float(row[2]), float(row[3]), float(row[6])))
        except Exception:
            pass


def stat(a, b, idx, busy):
    v = [(p, c) for (t, i, p, c, u) in S if a <= t <= b and i == idx and (u >= 50 or not busy)]
    if not v:
        return None
    ps = sorted(p for p, _ in v)
    return dict(n=len(ps), mean=round(sum(ps) / len(ps), 1), p95=round(ps[int(0.95 * (len(ps) - 1))], 1),
                max=round(ps[-1], 1), sm=round(sum(c for _, c in v) / len(v)))


rows = []
with open(f"{R}/marks.csv") as f:
    for m in csv.DictReader(f):
        a, b = float(m["t_start"]), float(m["t_end"])
        rec = {"arm": m["arm"], "row": m["row"], "dur_s": round(b - a, 1)}
        for i in (0, 1):
            rec[f"gpu{i}_win"] = stat(a, b, i, False)
            rec[f"gpu{i}_busy"] = stat(a, b, i, True)
        p = f"{R}/probe-{m['arm']}-{m['row']}.out"
        if os.path.exists(p):
            txt = open(p, errors="ignore").read()
            for k in ("ss_agg_tps_median", "ss_per_stream_tps_median", "ttft_s_median", "accept_per_draft_median"):
                mm = re.search(rf'"{k}"\s*:\s*([0-9.]+)', txt)
                if mm:
                    rec[k] = float(mm.group(1))
        rows.append(rec)
json.dump(rows, open(f"{R}/summary-busy.json", "w"), indent=1)

arms = sorted({r["arm"] for r in rows}, key=lambda a: [r["arm"] for r in rows].index(a))
names = sorted({r["row"] for r in rows}, key=lambda n: [r["row"] for r in rows].index(n))
base = arms[0]


def get(arm, name):
    return next((r for r in rows if r["arm"] == arm and r["row"] == name), None)


for metric, label, better_up in (("ss_agg_tps_median", "aggregate decode tok/s", True),
                                 ("ttft_s_median", "TTFT s", False)):
    print(f"\n== {label} (delta vs {base}) ==")
    print(f"{'row':<14}" + "".join(f"{a:>26}" for a in arms))
    for name in names:
        cells = []
        b = (get(base, name) or {}).get(metric)
        for a in arms:
            v = (get(a, name) or {}).get(metric)
            if v is None:
                cells.append(f"{'-':>26}")
            elif a == base or not b:
                cells.append(f"{v:>26.2f}")
            else:
                cells.append(f"{v:>18.2f} {(v - b) / b * 100:>+6.1f}%")
        print(f"{name:<14}" + "".join(cells))

print("\n== power draw while busy (util >= 50), mean / p95 W, and mean SM MHz ==")
print(f"{'arm':<16}{'row':<14}{'GPU0 busy':>16}{'GPU1 busy':>16}{'GPU0 win':>16}{'GPU1 win':>16}{'sm 0/1':>12}")
for r in rows:
    def f(k):
        d = r.get(k)
        return f"{d['mean']:.0f}/{d['p95']:.0f}" if d else "-"
    b0 = r.get("gpu0_busy") or {}
    b1 = r.get("gpu1_busy") or {}
    print(f"{r['arm']:<16}{r['row']:<14}{f('gpu0_busy'):>16}{f('gpu1_busy'):>16}{f('gpu0_win'):>16}{f('gpu1_win'):>16}"
          f"{str(b0.get('sm','-')) + '/' + str(b1.get('sm','-')):>12}")
