#!/usr/bin/env python3
"""Print the per-run generation t/s values of a llama-benchy --format json file, one line per concurrency.
The box's single-stream decode is bimodal (R163c), so a mean±std hides the two modes; the run list shows them.
usage: benchy_runs.py FILE [TAG]"""
import json, sys
d = json.load(open(sys.argv[1])); tag = sys.argv[2] if len(sys.argv) > 2 else ""
for b in d.get("benchmarks", []):
    vals = [round(v, 1) for v in b.get("tg_throughput", {}).get("values", [])]
    print(f"[{tag}] per-run c={b.get('concurrency')} tg_tps={vals} max={max(vals) if vals else 0}", flush=True)
