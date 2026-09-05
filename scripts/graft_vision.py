#!/usr/bin/env python3
"""R199 (2026-09-05): give a text-only ModelOpt NVFP4 export its vision tower back so vLLM's Qwen3.5 wrapper can load it.

Shiftedx/Qwopus3.8-27B-Flash-NVFP4-MTP ships no `model.visual.*` tensors and no `model.visual*` exclusion, so vLLM builds the
vision tower with the NVFP4 linear method and dies in create_weights ("in features size is not multiple of 16"); with the
exclusion alone it would die on the missing weights instead. Every ModelOpt checkpoint that boots on this route (gittensor,
natfii, inferact, heretic) carries `model.visual*` in exclude_modules AND the bf16 vision weights. This script builds a new
checkpoint directory: hard links to every file of the quant, plus one extra safetensors shard holding the bf16 source's
`model.visual.*` tensors copied byte-for-byte (no torch/numpy needed), the index merged, `vision_config` taken from the bf16
source (the weights must match it), and `model.visual*` appended to both exclusion lists (config.json `ignore`,
hf_quant_config.json `exclude_modules`). The language model, MTP head and every quantized tensor are untouched.

    graft_vision.py --quant DIR --bf16 DIR --out DIR
"""
import argparse, json, os, struct, sys

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quant", required=True); ap.add_argument("--bf16", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--prefix", default="model.visual."); ap.add_argument("--pattern", default="model.visual*")
    a = ap.parse_args()
    if os.path.exists(a.out): sys.exit(f"refusing to overwrite {a.out}")
    qidx = json.load(open(os.path.join(a.quant, "model.safetensors.index.json")))
    if any(k.startswith(a.prefix) for k in qidx["weight_map"]): sys.exit("quant already has vision tensors; nothing to graft")
    bidx = json.load(open(os.path.join(a.bf16, "model.safetensors.index.json")))["weight_map"]
    vis = sorted(k for k in bidx if k.startswith(a.prefix))
    if not vis: sys.exit("bf16 source has no vision tensors")
    os.makedirs(a.out)
    for name in os.listdir(a.quant):
        if name in ("config.json", "hf_quant_config.json", "model.safetensors.index.json") or name.startswith("."): continue
        # hard links, not symlinks: the serving container mounts ONLY the output dir, so a symlink into the quant dir dangles inside it
        # (R199 second run: dangling tokenizer files -> "ReasoningConfig: failed to tokenize reasoning strings").
        src, dst = os.path.join(a.quant, name), os.path.join(a.out, name)
        try: os.link(src, dst)
        except OSError: import shutil; shutil.copy2(src, dst)
    # gather tensors per source shard, write one new shard
    shard_name = "model-visual-from-bf16.safetensors"; header = {"__metadata__": {"format": "pt", "grafted_from": os.path.basename(a.bf16)}}
    chunks = []; off = 0
    for shard in sorted({bidx[k] for k in vis}):
        h, data_start = read_header(os.path.join(a.bf16, shard))
        for k in vis:
            if bidx[k] != shard: continue
            e = h[k]; s, t = e["data_offsets"]; n = t - s
            header[k] = {"dtype": e["dtype"], "shape": e["shape"], "data_offsets": [off, off + n]}
            chunks.append((os.path.join(a.bf16, shard), data_start + s, n)); off += n
    hb = json.dumps(header, separators=(",", ":")).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(os.path.join(a.out, shard_name), "wb") as w:
        w.write(struct.pack("<Q", len(hb))); w.write(hb)
        for path, start, n in chunks:
            with open(path, "rb") as r:
                r.seek(start)
                while n:
                    b = r.read(min(n, 64 << 20)); w.write(b); n -= len(b)
    wm = dict(qidx["weight_map"]); wm.update({k: shard_name for k in vis})
    meta = dict(qidx.get("metadata", {})); meta["total_size"] = meta.get("total_size", 0) + off
    json.dump({"metadata": meta, "weight_map": wm}, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
    cfg = json.load(open(os.path.join(a.quant, "config.json"))); bcfg = json.load(open(os.path.join(a.bf16, "config.json")))
    diff = {k: (cfg.get("vision_config", {}).get(k), bcfg["vision_config"].get(k)) for k in set(cfg.get("vision_config", {})) | set(bcfg["vision_config"]) if cfg.get("vision_config", {}).get(k) != bcfg["vision_config"].get(k)}
    cfg["vision_config"] = bcfg["vision_config"]
    ig = cfg.setdefault("quantization_config", {}).setdefault("ignore", [])
    if a.pattern not in ig: ig.append(a.pattern)
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    hq_path = os.path.join(a.quant, "hf_quant_config.json")
    if os.path.exists(hq_path):
        hq = json.load(open(hq_path)); ex = hq.setdefault("quantization", {}).setdefault("exclude_modules", [])
        if a.pattern not in ex: ex.append(a.pattern)
        json.dump(hq, open(os.path.join(a.out, "hf_quant_config.json"), "w"), indent=2)
    print(f"grafted {len(vis)} tensors ({off/1e9:.2f} GB) into {a.out}/{shard_name}; index {len(qidx['weight_map'])} -> {len(wm)} keys; vision_config diff (quant, bf16): {diff}; ignore += {a.pattern}")

if __name__ == "__main__":
    main()
