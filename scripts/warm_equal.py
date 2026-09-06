#!/usr/bin/env python3
"""T=0 cold-vs-warm completion equality (R205).

Sends N deterministic prompts of ~CTX tokens twice each at temperature 0 and compares the completions. The first send
prefills cold; the second must be served from the prefix cache (GPU hit, or the CPU/disk tier after --flood). The
needle gate only checks that an answer is *found*; this checks that a prefix-cache-served continuation is the same
continuation the cold prefill produced, i.e. that the reused Mamba checkpoints are the right state. Any mismatch is a
FAIL: at T=0 on one engine the two sends are the same forward pass on the same tokens.

RESULT line: {"equal": k, "n": N, "hits_delta": [...], "per_prompt": [...]}.
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
    rng = random.Random(seed)
    words = []
    while len(words) < int(ctx * 0.75):
        words.extend(rng.sample(WORDS, len(WORDS)))
        words.append(f"entry-{rng.randint(1000, 9999)}.")
    body = " ".join(words)
    return f"Ledger {seed}:\n{body}\n\nSummarize in one sentence which fields the clerks record, then list three entry numbers that appear above:"


def complete(url, model, prompt, max_tokens):
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "seed": 1,
                       "logprobs": 1}).encode()
    req = urllib.request.Request(url + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.load(r)
    c = j["choices"][0]
    toks = c.get("logprobs", {}).get("tokens") or []
    return c["text"], toks, j["usage"]["prompt_tokens"], time.time() - t0


def flood(url, model, n, ctx):
    for i in range(n):
        complete(url, model, make_prompt(10_000 + i, ctx), 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--ctx", type=int, default=6000)
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--flood", type=int, default=0, help="between sends: N fresh prompts to evict GPU blocks")
    p.add_argument("--flood-ctx", type=int, default=90000)
    a = p.parse_args()
    per, hits_delta, equal = [], [], 0
    prompts = [make_prompt(500 + i, a.ctx) for i in range(a.n)]
    cold = []
    for i, pr in enumerate(prompts):
        text, toks, ptok, dt = complete(a.url, a.model, pr, a.max_tokens)
        cold.append((text, toks, ptok, dt))
    if a.flood:
        flood(a.url, a.model, a.flood, a.flood_ctx)
    for i, pr in enumerate(prompts):
        q0, h0 = metrics(a.url)
        text, toks, ptok, dt = complete(a.url, a.model, pr, a.max_tokens)
        q1, h1 = metrics(a.url)
        same = text == cold[i][0]
        equal += int(same)
        hits_delta.append(h1 - h0)
        first_diff = next((k for k, (x, y) in enumerate(zip(cold[i][1], toks)) if x != y), None) if not same else None
        per.append({"i": i, "prompt_tokens": ptok, "cold_s": round(cold[i][3], 3), "warm_s": round(dt, 3),
                    "hits_delta": h1 - h0, "queries_delta": q1 - q0, "equal": same, "first_diff_tok": first_diff,
                    "cold": cold[i][0][:80], "warm": text[:80]})
    print("RESULT " + json.dumps({"equal": equal, "n": a.n, "ctx": a.ctx, "flood": a.flood, "hits_delta": hits_delta,
                                  "per_prompt": per}))
    print("PASS" if equal == a.n else "FAIL")


if __name__ == "__main__":
    main()
