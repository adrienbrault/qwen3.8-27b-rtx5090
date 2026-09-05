#!/usr/bin/env python3
"""R201 (2026-09-06): ModelOpt MIXED_PRECISION checkpoints carry the per-layer algo map twice — config.json
quantization_config.quantized_layers (what vLLM reads first) and hf_quant_config.json quantization.quantized_layers.
Kearuga's config.json is missing the 10 FP8-materialized boundary MLPs that hf_quant_config.json and the tensors
declare, so vLLM builds them unquantized and the weight loader dies on their weight_scale/input_scale tensors.
This writes a sibling dir (hard links, fallback copy) whose config.json quantized_layers is the union of the two.
Usage: merge_quantized_layers.py --src DIR --out DIR
"""
import argparse, json, os, shutil

ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
src, out = os.path.abspath(a.src), os.path.abspath(a.out)
cfg = json.load(open(os.path.join(src, "config.json")))
hq = json.load(open(os.path.join(src, "hf_quant_config.json")))["quantization"]
qc = cfg["quantization_config"]
assert qc.get("quant_method") == "modelopt" and qc.get("quant_algo") == "MIXED_PRECISION", qc.get("quant_algo")
before = dict(qc.get("quantized_layers", {}))
missing = {k: v for k, v in hq["quantized_layers"].items() if k not in before}
conflict = {k for k, v in hq["quantized_layers"].items() if k in before and before[k]["quant_algo"] != v["quant_algo"]}
assert not conflict, f"algo conflicts between the two files: {sorted(conflict)[:5]}"
qc["quantized_layers"] = {**before, **missing}
os.makedirs(out, exist_ok=True)
for name in sorted(os.listdir(src)):
    s, d = os.path.join(src, name), os.path.join(out, name)
    if name == "config.json" or os.path.exists(d) or os.path.isdir(s):
        continue
    try:
        os.link(s, d)
    except OSError:
        shutil.copy2(s, d)
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
print(f"config.json quantized_layers {len(before)} -> {len(qc['quantized_layers'])}; added {len(missing)}:")
for k, v in sorted(missing.items()):
    print(f"  {k}: {v['quant_algo']}")
