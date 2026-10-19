#!/usr/bin/env python3
"""Decode-path fidelity probe (R107b): the prefill-logprob ruler is BLIND to decode-only kernels
(prompt_logprobs + max_tokens=1 never runs a decode step), so this measures the decode read path
directly: T=0 greedy generations with per-token logprobs, compared across two arms of the same
engine (e.g. XQA decode vs FA2 decode via VLLM_SM12X_NVFP4_XQA). Two numerically-correct kernels
agree token-for-token except occasional near-tie flips with tiny |Δlogprob|; a wrong KV read
diverges early with large deltas.

  decode_fidelity.py run --url .. --corpus corpus.jsonl --out run-A.jsonl [--chunks 20] [--ctx 30000]
  decode_fidelity.py compare A.jsonl B.jsonl
"""
import argparse, json, random, urllib.request

FILLER = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra "
          "tango uniform victor whiskey xray yankee zulu river mountain harbor lantern copper meadow orbit velvet canyon ember ").split()

def run(a):
    chunks = [json.loads(l) for l in open(a.corpus)][:a.chunks]
    rng = random.Random("r107b")
    pad = (" ".join(rng.choice(FILLER) for _ in range(int(a.ctx / 1.3))) + "\n\nIgnore the words above.\n\n") if a.ctx else ""
    with open(a.out, "w") as out:
        for c in chunks:
            body = {"model": a.model, "prompt": pad + c["text"], "max_tokens": a.tokens, "temperature": 0, "logprobs": 2}
            r = urllib.request.Request(a.url + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            d = json.load(urllib.request.urlopen(r, timeout=1800))
            ch = d["choices"][0]
            # A completion with no generated token comes back with logprobs None (TabbyAPI; R668 crashed on it at
            # nvfp4/4,4). Record it as an empty generation, with its finish reason and text, never as a crash.
            lp = ch.get("logprobs") or {"tokens": [], "token_logprobs": []}
            out.write(json.dumps({"id": c["id"], "bucket": c.get("bucket"), "ctx": a.ctx,
                                  "tokens": lp["tokens"], "logprobs": lp["token_logprobs"],
                                  # top-k alternatives per position: lets a comparison tell a near-tie flip from a real
                                  # divergence (ba4-lean, 2026-10-19). Absent on servers that do not return them.
                                  "top": lp.get("top_logprobs"),
                                  "finish_reason": ch.get("finish_reason"), "text": ch.get("text", "")[:200]}) + "\n")
            print(f"chunk {c['id']} ok ({len(lp['tokens'])} tokens, finish {ch.get('finish_reason')})", flush=True)

def compare(a):
    A = {(j["id"], j["ctx"]): j for j in map(json.loads, open(a.runs[0]))}
    B = {(j["id"], j["ctx"]): j for j in map(json.loads, open(a.runs[1]))}
    tot = agree = 0; first_div = []; dlp = []
    for k in sorted(set(A) & set(B)):
        ta, tb = A[k]["tokens"], B[k]["tokens"]
        la, lb = A[k]["logprobs"], B[k]["logprobs"]
        n = min(len(ta), len(tb)); div = None
        for i in range(n):
            if ta[i] != tb[i]: div = i; break
            if la[i] is not None and lb[i] is not None: dlp.append(abs(la[i] - lb[i]))
        if div is None and len(ta) != len(tb): div = n   # one side stopped early (or generated nothing): not agreement
        tot += 1
        if div is None: agree += 1
        else: first_div.append(div)
    dlp.sort()
    med = dlp[len(dlp)//2] if dlp else None
    p99 = dlp[int(len(dlp)*0.99)] if dlp else None
    missing = len(set(A) ^ set(B))   # chunks present on one side only: a crashed or partial run
    print(json.dumps({"chunks": tot, "missing_one_side": missing, "fully_agreeing": agree, "diverged": len(first_div),
                      "first_divergence_positions": sorted(first_div)[:10],
                      "median_abs_dlogprob_on_agreed": round(med, 5) if med is not None else None,
                      "p99_abs_dlogprob_on_agreed": round(p99, 5) if p99 is not None else None}))

def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--url", required=True); r.add_argument("--model", default="qwen3.8-27b")
    r.add_argument("--corpus", required=True); r.add_argument("--out", required=True)
    r.add_argument("--chunks", type=int, default=20); r.add_argument("--ctx", type=int, default=0)
    r.add_argument("--tokens", type=int, default=128); r.set_defaults(f=run)
    c = sp.add_parser("compare"); c.add_argument("runs", nargs=2); c.set_defaults(f=compare)
    a = ap.parse_args(); a.f(a)

if __name__ == "__main__":
    main()
