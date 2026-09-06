#!/usr/bin/env python3
"""R205c: re-send one deterministic prompt K times and report per-send prefix-cache hits, ttft and latency.

Knobs that R205 found to matter for the live MTP zero-hit symptom are exposed so one boot can contrast them:
  --api chat|completions   (needle_depth.py uses chat + long answers and its 2nd re-ask HIT 129,536/131,245;
                            warm-revisit.py uses completions + max_tokens 8 and never hits)
  --max-tokens N           answer length of every send
  --gap S                  seconds between sends (needle re-asks wait 15 s; warm-revisit waits ~0)
RESULT line: {"api":..,"ctx":..,"max_tokens":..,"gap":..,"sends":[{"i","prompt_tokens","hits_delta","queries_delta","ttft_s","total_s"}]}.
"""
import argparse
import json
import random
import time
import urllib.request

WORDS = ("the archive keeps a ledger of every shipment that crossed the northern pass before the thaw, and the clerks "
         "record the weight, the origin, the carrier, the toll paid, and the seal that closed each crate").split()


def metrics(url):
    q = h = 0.0
    with urllib.request.urlopen(url + "/metrics", timeout=10) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("vllm:prefix_cache_queries_total"):
                q += float(line.rsplit(" ", 1)[1])
            elif line.startswith("vllm:prefix_cache_hits_total"):
                h += float(line.rsplit(" ", 1)[1])
    return q, h


def make_prompt(seed, ctx):
    rng = random.Random(f"reask-{seed}")
    words = []
    while len(words) < int(ctx * 0.75):
        words.extend(rng.sample(WORDS, len(WORDS)))
        words.append(f"entry-{rng.randint(1000, 9999)}.")
    return f"Ledger {seed}:\n" + " ".join(words) + "\n\nList three entry numbers that appear above, then summarize the ledger in two sentences:"


def send(url, model, api, prompt, max_tokens):
    if api == "chat":
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                "temperature": 0.0, "stream": True}
        path = "/v1/chat/completions"
    else:
        body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "stream": True}
        path = "/v1/completions"
    body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    ptok = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            j = json.loads(line[5:])
            if ttft is None and j.get("choices"):
                ttft = time.time() - t0
            if j.get("usage"):
                ptok = j["usage"].get("prompt_tokens")
    return ptok, ttft, time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api", default="completions", choices=["chat", "completions"])
    p.add_argument("--ctx", type=int, default=6000)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--gap", type=float, default=0.0)
    p.add_argument("--sends", type=int, default=2)
    p.add_argument("--seed", default="a")
    a = p.parse_args()
    prompt = make_prompt(a.seed, a.ctx)
    out = []
    for i in range(a.sends):
        if i and a.gap:
            time.sleep(a.gap)
        q0, h0 = metrics(a.url)
        ptok, ttft, tot = send(a.url, a.model, a.api, prompt, a.max_tokens)
        time.sleep(1)
        q1, h1 = metrics(a.url)
        out.append({"i": i, "prompt_tokens": ptok, "hits_delta": h1 - h0, "queries_delta": q1 - q0,
                    "ttft_s": None if ttft is None else round(ttft, 3), "total_s": round(tot, 3)})
    print("RESULT " + json.dumps({"api": a.api, "ctx": a.ctx, "max_tokens": a.max_tokens, "gap": a.gap, "seed": a.seed,
                                  "sends": out}))


if __name__ == "__main__":
    main()
