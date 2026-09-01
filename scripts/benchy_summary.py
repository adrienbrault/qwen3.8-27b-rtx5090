#!/usr/bin/env python3
"""Print one RESULT line per (concurrency, pp, tg) from a llama-benchy --format json file.
tg_throughput = generation tokens/s (tokenizer-counted); std across runs is the honest noise."""
import json, sys
d = json.load(open(sys.argv[1])); tag = sys.argv[2] if len(sys.argv) > 2 else ""
for b in d.get("benchmarks", []):
    tg = b.get("tg_throughput", {}); pk = b.get("peak_throughput", {}); pp = b.get("pp_throughput", {}); tt = b.get("ttfr", {})
    n = len(tg.get("values", []))
    print(f"[{tag}] RESULT c={b.get('concurrency')} pp={b.get('prompt_size')} tg={b.get('response_size')} "
          f"tg_tps={tg.get('mean',0):.1f}+-{tg.get('std',0):.1f} (n={n}) peak={pk.get('mean',0):.1f} "
          f"pp_tps={pp.get('mean',0):.0f} ttfr_ms={tt.get('mean',0):.0f}", flush=True)
