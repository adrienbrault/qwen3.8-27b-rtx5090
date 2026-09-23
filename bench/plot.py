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
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
OUT = ROOT / "docs" / "img"
R675 = RESULTS / "2026-09-23-r675-27b-curves"
R206C = RESULTS / "2026-09-06-r206c-mtp-c32c64-v2"

CODE, PROSE, PREFILL = "#0969da", "#cf222e", "#8250df"
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


def decode_ss_dir(path, pattern):
    """One decode_ss summary per file (a concurrency sweep writes one line per c; a single-shape
    run writes one line). Returns {c: (agg_median, per_stream_median)}."""
    out = {}
    for p in sorted(path.glob(pattern)):
        out.update(decode_ss(p))
    return out


def decode_figure(rates, name, caption, label):
    """Two panels, not twin axes: the aggregate and per-stream lines cross between 8 and 12 streams
    and their labels would print on top of one another. `label` names the measurement for the
    console output; `rates` maps kind -> {c: (agg, per_stream)}."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for kind, color, dy in (("code", CODE, 7), ("prose", PROSE, -14)):
        conc = sorted(rates[kind])
        agg = [rates[kind][c][0] for c in conc]
        per = [rates[kind][c][1] for c in conc]
        ax.plot(conc, agg, marker="o", color=color, linewidth=2, label=kind)
        ax2.plot(conc, per, marker="s", markersize=4, color=color, linewidth=1.8, label=kind)
        annotate(ax, conc, agg, color, dy=dy)
        annotate(ax2, conc, per, color, dy=dy)
        ax.set_xticks(conc)
        ax2.set_xticks(conc)
    ax.set_title("Decode rate, all streams")
    ax.set_ylabel("tokens per second, sum of streams")
    ax.set_ylim(0, max(rates[k][c][0] for k in rates for c in rates[k]) * 1.2)
    ax2.set_title("Decode rate, one stream")
    ax2.set_ylabel("tokens per second, per stream")
    ax2.set_ylim(0, max(rates[k][c][1] for k in rates for c in rates[k]) * 1.3)
    for a in (ax, ax2):
        a.set_xlabel("concurrent streams")
        a.grid(axis="y", color="#eaeef2")
        a.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax2.legend(frameon=False, fontsize=9, loc="lower left")
    print(f"{label}: aggregate", {k: [round(rates[k][c][0]) for c in sorted(rates[k])] for k in rates})
    print(f"{label}: per stream", {k: [round(rates[k][c][1]) for c in sorted(rates[k])] for k in rates})
    save(fig, name, caption)


def figure_decode_scaling():
    rates = {k: decode_ss(R675 / f"decode-{k}.jsonl") for k in ("code", "prose")}
    decode_figure(rates, "decode-scaling.svg",
                  "Decode rate against concurrency, aggregate and per stream", "decode scaling (R675)")


def figure_decode_scaling_64():
    """R206c ran the same probe on a seq-64 boot (2026-09-06, RedHatAI checkpoint, 13.98 GB pin) — a
    different served configuration, so it is its own figure, not a series on the R675 curve."""
    rates = {k: decode_ss_dir(R206C, f"decode-M3-{k}-c*.jsonl") for k in ("code", "prose")}
    decode_figure(rates, "decode-scaling-64.svg",
                  "Decode rate against concurrency on the seq-64 boot, aggregate and per stream",
                  "decode scaling seq-64 (R206c)")


def prefill_points():
    """probes/kv_capacity_probe.py with --conc 1 --tokens 1: one cold request per line, [latency s, prompt tokens]
    as counted by the server. With a single output token the latency is the time to first token."""
    rows = collections.defaultdict(list)
    for line in open(R675 / "prefill.jsonl"):
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
        for line in open(R675 / f"decode-{kind}.jsonl"):
            s = json.loads(line)["summary"]
            if s["c"] == 1:
                out[kind].append((0, s["ss_per_stream_tps_median"]))
    for p in sorted(R675.glob("decode-*-c1-*k.jsonl")):
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
