#!/usr/bin/env python3
"""R161: multi-image prompt probe against an OpenAI-compatible vLLM endpoint (stdlib only).

Usage: mm_probe.py BASE_URL IMG_DIR N LABEL OUT_JSON [--no-decode]

Runs, in order, against N images from IMG_DIR (shot01.png ... , codes.txt holds the 4-digit code per image):
  1. cold: N images + "list the codes"  -> TTFT, total, prompt tokens, correctness vs codes.txt
  2. warm: identical request           -> TTFT (full prefix hit)
  3. plus1: N+1 images                 -> TTFT (incremental: only the new image should be encoded)
  4. decode-img: N images + essay ask, 512 tokens -> decode t/s after first token, spec acceptance
  5. decode-txt: text-only prompt of similar token count + same essay ask -> same
Each step records vLLM /metrics deltas: prompt tokens, prefix-cache hits/queries, external hits, spec-decode counters.
"""
import base64, json, sys, time, urllib.request

BASE, IMGDIR, N, LABEL, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
DECODE = "--no-decode" not in sys.argv
MODEL = "qwen3.8-27b"


def metrics():
    m = {}
    for ln in urllib.request.urlopen(BASE.rsplit("/v1", 1)[0] + "/metrics", timeout=20).read().decode().splitlines():
        if ln.startswith("#"):
            continue
        k, _, v = ln.rpartition(" ")
        try:
            m[k.split("{")[0]] = m.get(k.split("{")[0], 0.0) + float(v)
        except ValueError:
            pass
    return m


KEYS = {
    "prompt": "vllm:prompt_tokens_total", "gen": "vllm:generation_tokens_total",
    "pc_q": "vllm:prefix_cache_queries_total", "pc_h": "vllm:prefix_cache_hits_total",
    "ext_q": "vllm:external_prefix_cache_queries_total", "ext_h": "vllm:external_prefix_cache_hits_total",
    "drafts": "vllm:spec_decode_num_drafts_total", "draft_tok": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total", "preempt": "vllm:num_preemptions_total",
}


def delta(a, b):
    return {k: round(b.get(v, 0) - a.get(v, 0), 3) for k, v in KEYS.items()}


def img_part(i):
    with open(f"{IMGDIR}/shot{i:02d}.png", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def chat(content, max_tokens):
    """Streams; returns dict(ttft, total, gen_tokens, text, usage)."""
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": content}],
                       "max_tokens": max_tokens, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body, headers={"content-type": "application/json"})
    t0 = time.time(); ttft = None; first = None; text = []; reasoning = 0; usage = None; n = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                    if ttft is None:
                        ttft = time.time() - t0; first = time.time()
                    n += 1
                    if d.get("content"):
                        text.append(d["content"])
                    else:
                        reasoning += 1
    total = time.time() - t0
    gen = usage["completion_tokens"] if usage else n
    return {"ttft": round(ttft or total, 3), "total": round(total, 3), "gen_tokens": gen,
            "decode_tps": round(gen / max(1e-6, (time.time() - first)), 1) if first else None,
            "text": "".join(text)[:400], "usage": usage}


codes = [int(x) for x in open(f"{IMGDIR}/codes.txt").read().split()]
res = {"label": LABEL, "n_images": N, "steps": {}}


def step(name, content, max_tokens):
    m0 = metrics(); r = chat(content, max_tokens); m1 = metrics()
    r["metrics"] = delta(m0, m1)
    acc = r["metrics"]
    if acc["draft_tok"]:
        r["accept_per_draft_tok"] = round(acc["accepted"] / acc["draft_tok"], 3)
    res["steps"][name] = r
    print(f"[{LABEL}] {name}: ttft {r['ttft']}s total {r['total']}s prompt_tok {(r['usage'] or {}).get('prompt_tokens')} "
          f"gen {r['gen_tokens']} decode {r['decode_tps']} t/s | pc_hit {acc['pc_h']:.0f}/{acc['pc_q']:.0f} ext_hit {acc['ext_h']:.0f}/{acc['ext_q']:.0f} "
          f"acc/draft {r.get('accept_per_draft_tok')}", flush=True)
    return r


ask = ("Each image shows a page with one 4-digit code in a blue box. List the codes in image order as a "
       "comma-separated list, nothing else.")
parts = [img_part(i) for i in range(1, N + 1)] + [{"type": "text", "text": ask}]
r = step("cold", parts, 300)
found = sum(1 for c in codes[:N] if str(c) in r["text"]); res["cold_codes_found"] = found
print(f"[{LABEL}] codes found {found}/{N}: {r['text'][:120]!r}")
step("warm", parts, 300)
step("plus1", [img_part(i) for i in range(1, N + 2)] + [{"type": "text", "text": ask}], 300)
if DECODE:
    essay = "Write a 600-word essay on the history of pinball machines, plain prose, no lists."
    step("decode_img", [img_part(i) for i in range(1, N + 1)] + [{"type": "text", "text": essay}], 512)
    ptok = (r["usage"] or {}).get("prompt_tokens", 20000)
    filler = ("The quick brown fox jumps over the lazy dog. " * 8 + "\n") * max(1, ptok // 90)
    step("decode_txt", [{"type": "text", "text": filler + "\n\n" + essay}], 512)
json.dump(res, open(OUT, "w"), indent=1)
print(f"[{LABEL}] wrote {OUT}")
