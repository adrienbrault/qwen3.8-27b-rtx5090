#!/usr/bin/env python3
"""Warm-revisit identity: does a prefix-cache hit change the greedy continuation?

For N fresh prompts (filler of --ctx tokens + a question), send each twice, cold then warm, greedy (T 0), --max-tokens
completion tokens, and compare the two texts: identical, or the character index where they first differ. Written for the
disable_eagle_block_drop screen (2026-10-20): the flag keeps the last full-attention block on a prefix hit, so the MTP
drafter starts from a slot written with the producer's lookahead token; verification keeps the target's tokens, so warm
greedy text should equal cold as often as it does without the flag.
RESULT line: {"ctx", "n", "identical", "first_diff": [...], "ttft_cold": [...], "ttft_warm": [...]}.
"""
import argparse
import json
import random
import time
import urllib.request

WORDS = ("the archive keeps a ledger of every shipment that crossed the northern pass before the thaw, and the clerks "
         "record the weight, the origin, the carrier, the toll paid, and the seal that closed each crate").split()


def send(url, model, prompt, max_tokens):
    # chat + thinking off: a raw /v1/completions prompt makes this chat checkpoint stop at once (empty text, 2026-10-20).
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = []; raw = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if len(raw) < 6:
                raw.append(line[:300])
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ch = json.loads(line[6:]).get("choices") or [{}]
            delta = ch[0].get("delta") or {}
            piece = (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
            if piece and ttft is None:
                ttft = time.time() - t0
            text.append(piece)
    if ttft is None:
        raise SystemExit("no completion text; first stream lines: " + json.dumps(raw))
    return "".join(text), ttft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=6000); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=256); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    same, diffs, tc, tw = 0, [], [], []
    for i in range(a.n):
        rng = random.Random(a.seed * 1000 + i)
        filler = " ".join(rng.choice(WORDS) for _ in range(int(a.ctx * 0.75)))
        prompt = (f"Ledger {a.seed}-{i}.\n{filler}\n\nWrite a Python function that parses one ledger line of the form "
                  f"'weight,origin,carrier,toll' into a dict, with type hints and a docstring, then explain it briefly.\n")
        cold, t1 = send(a.url, a.model, prompt, a.max_tokens)
        warm, t2 = send(a.url, a.model, prompt, a.max_tokens)
        tc.append(round(t1, 3)); tw.append(round(t2, 3))
        if cold == warm:
            same += 1; diffs.append(None)
        else:
            diffs.append(next((k for k, (x, y) in enumerate(zip(cold, warm)) if x != y), min(len(cold), len(warm))))
    print("RESULT " + json.dumps({"ctx": a.ctx, "n": a.n, "identical": same, "first_diff": diffs, "ttft_cold": tc, "ttft_warm": tw}))


if __name__ == "__main__":
    main()
