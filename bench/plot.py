#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Draw the README's figures from the published raw records.

    uv run bench/plot.py            # writes docs/img/*.svg

Every figure reads `bench/results/<date>-<round>/`, so no figure can carry a number that is not
in this repository, and each prints what it drew so the values can be checked against the
round's write-up.
"""

import collections
import json
import math
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
OUT = ROOT / "docs" / "img"
R675 = RESULTS / "2026-09-23-r675-27b-curves"
# r27-as-curves (2026-10-16): R675's instruments, shapes and runs re-measured on the served v0.31.0 launcher (A_S). The
# decode-scaling-16 and prefill.svg read it.
R27C = RESULTS / "2026-10-16-r27-as-curves"
# R793 (2026-09-28): `vllm bench serve` v0.30.0 on ShareGPT V3 and Spec-Bench against the same served launcher, a fresh
# boot per cell, passes A and B; drawn dashed over R675's decode curve.
R793 = RESULTS / "2026-09-28-r793-27b-std-bench-1608"
# R793's ShareGPT pair at 6 streams differed by 3.17 % (833.9 / 807.9 tok/s), above the round's 3 % rule. The rule for its
# re-run, fixed before the re-run was read: take the re-run's A/B mean if its spread is <= 3 % and the mean is within 3 %
# of R793 pass A. R793b: 829.8 / 835.7 tok/s, spread 0.71 %, mean 832.7 (0.14 % below pass A), so that cell comes from here.
R793B = RESULTS / "2026-09-28-r793b-27b-c6-1905"
STD_OVERRIDE = {("sharegpt", 6): (R793B, "r793b-27b-c6")}
STD_UNIT = "r793-27b-std-bench"
STD_CONC = (1, 2, 4, 6, 8, 12, 16)
STD_PASSES = ("A", "B")
# R794: the seq-64 figure, both lines from one boot configuration: the served launcher with the sequence limit at 64 (port
# 8029, 14.86 GB pin, capture cap 320), decode_ss on one boot and vllm bench serve with a fresh boot per cell, as R675 / R793.
R794 = RESULTS / "2026-09-28-r794-27b-seq64-2018"
# R794's ShareGPT pair at 16 streams differed by 3.29 % (1,464.8 / 1,513.7 tok/s), above the round's 3 % rule. The rule for
# its re-run, fixed before the re-run was read: take the re-run's A/B mean if its spread is <= 3 % and the mean lies within
# 3 % of R794's pair mean 1,489.2. R794b: 1,509.8 / 1,502.0 tok/s, spread 0.52 %, mean 1,505.9 (1.12 % above), so that cell
# comes from here.
R794B = RESULTS / "2026-09-28-r794b-27b-c16-2312"
R794_OVERRIDE = {("sharegpt", 16): (R794B, "r794b-27b-c16")}
R794_UNIT = "r794-27b-seq64"
R794_DEC_CONC = (1, 2, 4, 8, 16, 24, 32, 48, 64)
R794_STD_CONC = (1, 4, 8, 16, 32, 48, 64)
DEC_RUNS = 3

CODE, PROSE, PREFILL = "#0969da", "#cf222e", "#8250df"
SHAREGPT, SPECBENCH = "#1a7f37", "#9a6700"
STD_DATASETS = (("sharegpt", "ShareGPT V3", SHAREGPT), ("specbench", "Spec-Bench", SPECBENCH))
plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 10,
    "axes.edgecolor": "#d8dee4",
    "axes.labelcolor": "#57606a",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "xtick.color": "#57606a",
    "ytick.color": "#57606a",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
})


def save(fig, name, caption):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, format="svg", bbox_inches="tight", metadata={"Title": caption})
    plt.close(fig)


def annotate(ax, xs, ys, color, fmt="{:.0f}", dy=7):
    """dy places a series' labels above (positive) or below (negative) its markers, so two series that
    read within a few tokens per second of each other do not print on top of one another."""
    for x, y in zip(xs, ys):
        if y is None:
            continue
        ax.annotate(fmt.format(y), (x, y), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=8.5, color=color)


def decode_ss(path):
    """probes/decode_ss.py writes one line per concurrency: {"summary": {...}, "runs": [...]}. The summary's
    medians are over runs, each run's rate over the samples where every stream was decoding."""
    out = {}
    for line in open(path):
        s = json.loads(line)["summary"]
        out[s["c"]] = (s["ss_agg_tps_median"], s["ss_per_stream_tps_median"])
    return out


def decode_ss_exact(path, concs, runs=DEC_RUNS):
    """decode_ss() for a sweep that must hold exactly one summary per concurrency in `concs`, each over `runs` good
    runs. A missing file, a missing or repeated concurrency, or a shape with a failed run is an error, never a gap."""
    out = {}
    for line in open(path):
        s = json.loads(line)["summary"]
        if s["c"] in out:
            raise ValueError(f"{path}: concurrency {s['c']} appears twice")
        if s["runs"] != runs:
            raise ValueError(f"{path}: c{s['c']} has {s['runs']} good runs, want {runs}")
        out[s["c"]] = (s["ss_agg_tps_median"], s["ss_per_stream_tps_median"])
    if sorted(out) != sorted(concs):
        raise ValueError(f"{path}: concurrencies {sorted(out)}, want {sorted(concs)}")
    return out


def std_bench(base=R793, unit=STD_UNIT, concs=STD_CONC, override=STD_OVERRIDE):
    """A round's cells per (dataset, conc), each the mean of passes A and B; R793's defaults take ShareGPT at 6 streams
    from R793b (STD_OVERRIDE). Output tok/s is `vllm bench serve`'s output_throughput: all completion tokens over the run's wall
    time, prefill, time to first token and request turnover included. Per stream is 1 / median over requests of
    TPOT = (latency - TTFT) / (output tokens - 1), which includes the steps a request waits while other requests'
    prefill chunks run. The Flash-Next repository's figure uses the same two definitions. A missing cell file, a file
    from another round or cell, or a cell with a failed request is an error, never a gap."""
    cells, src = {}, {}
    for key, _, _ in STD_DATASETS:
        for c in concs:
            cbase, cunit = override.get((key, c), (base, unit))
            vals = []
            for ps in STD_PASSES:
                f = cbase / f"{ps}-{key}-c{c}.json"
                if not f.exists():
                    raise FileNotFoundError(f"{f} is missing; every standard-benchmark cell needs both passes")
                d = json.load(open(f))
                want = dict(unit=cunit, dataset=key, conc=str(c), tag=f"{ps}-{key}-c{c}")
                got = {k: str(d.get(k)) for k in want}
                if got != want:
                    raise ValueError(f"{f}: metadata {got}, want {want}")
                if d["failed"] != 0 or d["completed"] != d["num_prompts"]:
                    raise ValueError(f"{f}: {d['completed']} of {d['num_prompts']} completed, {d['failed']} failed")
                tpot = [(lat - ttft) / (n - 1) for lat, ttft, n in zip(d["latencies"], d["ttfts"], d["output_lens"]) if n > 1]
                vals.append((d["output_throughput"], 1.0 / st.median(tpot)))
            cells[(key, c)] = (st.mean(o for o, _ in vals), st.mean(p for _, p in vals))
            src[(key, c)] = cbase.name
    return cells, src, base.name


def place_labels(ax, series, first, xunit=1.0):
    """Value labels that print on no marker, line or other label. `series` is a list of (xs, ys, color, labelled) in
    drawing order: the solid code line is labelled at every point, the dashed lines from `first` streams, the prose
    line not at all (as in the Flash-Next figure). Each label tries six spots in order (above centred, above left,
    above right, then the same below) and takes the first whose box is clear of every series and of the labels already
    placed, inside the y axis on the left (the open right edge may take half a label). With none clear the label is
    left out; the value stays in the write-up's tables. Boxes are in data units: a label is ~5 % of the y range tall
    and ~0.3 streams wide per digit at this figure size on the 1-16 stream axis; `xunit` scales every x distance for
    an axis of another span (the x span over the 1-16 axis's)."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    xlo, xhi = ax.get_xlim()

    def interp(xs, ys, x):
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    spots = [(v, h) for v in ("above", "below") for h in ("center", "left", "right")]
    placed = []   # (x0, x1, y0, y1) boxes in data units
    for i, (xs, ys, color, labelled) in enumerate(series):
        for x, y in zip(xs, ys):
            if not labelled or (i and x < first):
                continue
            text = f"{y:.0f}"
            w = 0.3 * xunit * len(text)
            for v, h in spots:
                x0 = {"center": x - w / 2, "left": x - w - 0.1 * xunit, "right": x + 0.1 * xunit}[h]
                x1 = x0 + w
                y0, y1 = (y + 0.012 * span, y + 0.062 * span) if v == "above" else (y - 0.075 * span, y - 0.025 * span)
                grid = [x0 + k * (x1 - x0) / 6 for k in range(7)]
                hit = any(y0 - 0.01 * span <= interp(xs2, ys2, gx) <= y1 + 0.01 * span
                          for j, (xs2, ys2, _, _) in enumerate(series) for gx in grid
                          if j != i or abs(gx - x) > 0.15 * xunit)
                hit = hit or x0 < xlo or x1 > xhi + 0.6 * xunit or any(px0 < x1 and x0 < px1 and py0 < y1 and y0 < py1 for px0, px1, py0, py1 in placed)
                if not hit:
                    placed.append((x0, x1, y0, y1))
                    dx = {"center": 0, "left": -3, "right": 3}[h]
                    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, 3 if v == "above" else -11),
                                ha={"center": "center", "left": "right", "right": "left"}[h], fontsize=8.5, color=color)
                    break


R793_NOTE = ("Solid: decode_ss.py, all streams decoding, 1,024 forced tokens, one code and one prose prompt, 2026-09-23 "
             "(R675; memory clock offset not recorded).\n"
             "Dashed: vllm bench serve, closed loop, ShareGPT reference-reply lengths / 256 tokens, mean of passes A and B, "
             "memory clock offset +4500, 2026-09-28 (R793; ShareGPT at 6 streams from the R793b re-run).")


def std_figure(rates, std, name, caption, rounds=("R675", "R793"),
               title="Decode alone against the standard benchmark, served configuration", note=R793_NOTE, xpos=None, first=2):
    """Two panels, not twin axes (the aggregate and per-stream lines cross): a round's standard-benchmark cells dashed
    over decode_ss solid. `rounds` names the solid and dashed rounds in the legend and the console output. `xpos` maps a
    concurrency to its x position (None: the concurrency itself, ticks at the dashed round's levels); with it every
    concurrency of either line gets a tick labelled with its value, and the label boxes scale with the x span. `first`
    is the x position from which the dashed lines are labelled."""
    cells, src, base = std if std is not None else ({}, {}, None)   # None: decode alone only, no dashed round
    X = xpos or (lambda c: c)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.9))
    sconc = sorted({c for (_, c) in cells})
    series = ([], [])
    for kind, color in (("code", CODE), ("prose", PROSE)):
        conc = sorted(rates[kind])
        xs = [X(c) for c in conc]
        for a, idx, i in ((ax, 0, 0), (ax2, 1, 1)):
            ys = [rates[kind][c][idx] for c in conc]
            a.plot(xs, ys, marker="o" if idx == 0 else "s", markersize=6 if idx == 0 else 4, color=color,
                   linewidth=2, label=f"{kind}, decode alone ({rounds[0]})")
            series[i].append((xs, ys, color, kind == "code"))
    sxs = [X(c) for c in sconc]
    for key, dname, color in (STD_DATASETS if cells else ()):
        for a, idx, i in ((ax, 0, 0), (ax2, 1, 1)):
            ys = [cells[(key, c)][idx] for c in sconc]
            a.plot(sxs, ys, marker="s", markersize=4, color=color, linewidth=2, linestyle="--",
                   label=f"{dname}, vllm bench serve ({rounds[1]})")
            series[i].append((sxs, ys, color, True))
    ys_agg = [y for _, ys, _, _ in series[0] for y in ys]
    ys_per = [y for _, ys, _, _ in series[1] for y in ys]
    ax.set_ylim(0, max(ys_agg) * 1.2)
    ax2.set_ylim(0, max(ys_per) * 1.5)   # room for the four-entry legend above the lines
    ax.set_title("All streams")
    ax.set_ylabel("tokens per second, sum of streams" + ("\n(dashed: output tok/s, wall clock)" if cells else ""))
    ax2.set_title("One stream")
    ax2.set_ylabel("tokens per second, per stream" + ("\n(dashed: 1000 / TPOT p50)" if cells else ""))
    ticks = sorted(set(sconc) | {c for k in rates for c in rates[k]})
    for a in (ax, ax2):
        if xpos is None:
            a.set_xticks(sconc or ticks)
        else:
            a.set_xticks([X(c) for c in ticks], [str(c) for c in ticks])
        a.set_xlabel("concurrent streams")
        a.grid(axis="y", color="#eaeef2")
        a.set_axisbelow(True)
    # the label constants were set on the 1-16 stream axis (span 15 plus matplotlib's 5 % margins)
    xunit = 1.0 if xpos is None else (ax.get_xlim()[1] - ax.get_xlim()[0]) / (1.1 * 15)
    place_labels(ax, series[0], first=first, xunit=xunit)
    place_labels(ax2, series[1], first=first, xunit=xunit)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=7.5, color="#57606a")
    print(f"decode scaling ({rounds[0]}): aggregate", {k: [round(rates[k][c][0]) for c in sorted(rates[k])] for k in rates})
    print(f"decode scaling ({rounds[0]}): per stream", {k: [round(rates[k][c][1]) for c in sorted(rates[k])] for k in rates})
    print(f"standard benchmark (passes A/B mean) at {sconc}")
    for key, dname, _ in (STD_DATASETS if cells else ()):
        print(f"  {dname:11}  output tok/s {[round(cells[(key, c)][0], 1) for c in sconc]}"
              f"   per stream {[round(cells[(key, c)][1]) for c in sconc]}")
    if cells:
        print("  sources:", sorted({(k, c, v) for (k, c), v in src.items() if v != base}), "; every other cell", base)
    save(fig, name, caption)


def figure_decode_scaling():
    """The served figure: decode alone on the served v0.31.0 launcher (r27-as-curves). R793's standard-benchmark lines
    were measured on the v0.29.0rc2 launcher, so they stay out of this figure until R793 is re-run on the served one."""
    rates = {k: decode_ss(R27C / f"decode-{k}.jsonl") for k in ("code", "prose")}
    std_figure(rates, None, "decode-scaling-16.svg",
               "Decode rate against concurrency on the served 16-sequence boot (r27-as-curves), sum over streams and per stream",
               rounds=("r27-as-curves", ""), title="Decode alone, served configuration",
               note="decode_ss.py, all streams decoding, 1,024 forced tokens, one code and one prose prompt, three runs per "
                    "point (median), stock power, 2026-10-16, served v0.31.0 launcher (r27-as-curves).")


def figure_decode_scaling_64():
    """R794 measured both lines on the served launcher booted with the sequence limit at 64, a configuration other than
    the served 16 sequences, so it is its own figure: decode_ss on one boot, vllm bench serve with a fresh boot per cell.
    The x axis is log2: nine concurrencies from 1 to 64 would crowd the left of a linear axis."""
    rates = {k: decode_ss_exact(R794 / f"decode-{k}.jsonl", R794_DEC_CONC) for k in ("code", "prose")}
    std = std_bench(R794, R794_UNIT, R794_STD_CONC, R794_OVERRIDE)
    std_figure(rates, std, "decode-scaling-64.svg",
               "Decode rate against concurrency on the 64-sequence boot: decode alone (R794, solid) and vllm bench serve "
               "on ShareGPT V3 and Spec-Bench (R794, dashed), sum over streams and per stream",
               rounds=("R794", "R794"), title="Decode alone against the standard benchmark, 64-sequence boot",
               note=("Solid: decode_ss.py, all streams decoding, 1,024 forced tokens, one code and one prose prompt, one boot. "
                     "Dashed: vllm bench serve, closed loop, ShareGPT reference-reply lengths / 256 tokens,\n"
                     "mean of passes A and B, a fresh boot per cell. Both: the served launcher with the sequence limit at 64 "
                     f"(14.86 GB pin, graph capture capped at 320), memory clock offset +4500, {R794.name[:10]} (R794;\n"
                     "ShareGPT at 16 streams from the R794b re-run). Streams on a log scale."),
               xpos=math.log2, first=math.log2(4))


def prefill_points():
    """probes/kv_capacity_probe.py with --conc 1 --tokens 1: one cold request per line, [latency s, prompt tokens]
    as counted by the server. With a single output token the latency is the time to first token."""
    rows = collections.defaultdict(list)
    for line in open(R27C / "prefill.jsonl"):
        r = json.loads(line)
        for req in r["requests"]:
            if isinstance(req[0], (int, float)) and req[0] > 0:
                rows[r["ctx"]].append((req[1], req[1] / req[0]))
    keys = sorted(rows)
    return ([st.mean(t for t, _ in rows[k]) for k in keys], [st.mean(v for _, v in rows[k]) for k in keys])


def depth_decode():
    """decode_ss.py at one stream on top of N filler words' worth of context (--ctx N, N/1.3 words), plus the
    no-filler c1 point of the same boot. decode_ss does not record the prompt's token count, so these points sit
    on their own axis: the filler budget, not measured prompt tokens."""
    out = collections.defaultdict(list)
    for kind in ("code", "prose"):
        for line in open(R27C / f"decode-{kind}.jsonl"):
            s = json.loads(line)["summary"]
            if s["c"] == 1:
                out[kind].append((0, s["ss_per_stream_tps_median"]))
    for p in sorted(R27C.glob("decode-*-c1-*k.jsonl")):
        kind = p.name.split("-")[1]
        for line in open(p):
            s = json.loads(line)["summary"]
            out[kind].append((s["ctx"], s["ss_per_stream_tps_median"]))
    return {k: sorted(v) for k, v in out.items()}


def figure_prefill():
    toks, rate = prefill_points()
    depth = depth_decode()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax.plot(toks, rate, marker="o", color=PREFILL, linewidth=2, label="prefill rate")
    annotate(ax, toks, rate, PREFILL)
    ax.set_title("Cold prefill rate, one request")
    ax.set_xlabel("prompt tokens (server count)")
    ax.set_ylabel("prompt tokens per second")
    ax.set_ylim(0, max(rate) * 1.3)
    ax.set_xticks(toks, [f"{round(t / 1000)}k" for t in toks])
    ax.grid(axis="y", color="#eaeef2")
    ax.set_axisbelow(True)
    for kind, color, dy in (("code", CODE, 7), ("prose", PROSE, -14)):
        xs = [t for t, _ in depth.get(kind, [])]
        ys = [v for _, v in depth.get(kind, [])]
        ax2.plot(xs, ys, marker="s", markersize=4, color=color, linewidth=1.8, label=kind)
        annotate(ax2, xs, ys, color, dy=dy)
    ax2.set_title("Decode at depth, one stream")
    ax2.set_xlabel("filler context (decode_ss --ctx)")
    ax2.set_ylabel("tokens per second")
    ax2.set_ylim(0, max(v for d in depth.values() for _, v in d) * 1.3)
    xt = sorted({t for d in depth.values() for t, _ in d})
    ax2.set_xticks(xt, ["0" if t == 0 else f"{round(t / 1000)}k" for t in xt])
    ax2.grid(axis="y", color="#eaeef2")
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, fontsize=9, loc="lower left")
    print("prefill:", [round(v) for v in rate], "t/s at", [round(t) for t in toks], "tokens")
    print("decode at depth:", {k: [(round(t), round(v)) for t, v in d] for k, d in depth.items()})
    save(fig, "prefill.svg", "Cold prefill rate against prompt length, and decode rate at depth")


if __name__ == "__main__":
    figure_decode_scaling()
    figure_decode_scaling_64()
    figure_prefill()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.svg"))))
