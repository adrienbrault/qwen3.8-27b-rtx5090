#!/usr/bin/env python3
"""Rescore lm-eval gsm8k samples with a markdown/commas-tolerant extractor (last number inside the last **bold** span,
else last number; $ and thousands separators stripped). lm-eval's flexible-extract misses `**18**` and `70,000`.
  gsm8k_rescore.py <samples_gsm8k*.jsonl> ..."""
import json, re, sys
for path in sys.argv[1:]:
    seen, ok, n = set(), 0, 0
    for l in open(path):
        d = json.loads(l)
        if d["doc_id"] in seen: continue
        seen.add(d["doc_id"]); n += 1
        target = d["target"].split("####")[-1].strip().replace(",", "")
        raw = d["resps"][0][0] if d.get("resps") else ""
        clean = lambda s: s.replace("$", "").replace(",", "")
        # the model states its answer in **bold**; trailing context ("...for the 16 glasses") fools a last-number rule
        bold = [m for b in re.findall(r"\*\*(.+?)\*\*", raw, flags=re.S) for m in re.findall(r"-?\d+(?:\.\d+)?", clean(b))]
        nums = bold or re.findall(r"-?\d+(?:\.\d+)?", clean(raw.replace("**", "")))
        if nums:
            try:
                ok += int(abs(float(nums[-1]) - float(target)) < 1e-6)
            except ValueError: pass
    print(f"{path.split('/')[-3] if '/' in path else path}: {ok}/{n} = {100*ok/n:.1f}%  (±{100*(ok/n*(1-ok/n)/n)**0.5:.1f})")
