#!/usr/bin/env python3
"""Steady-state concurrent decode probe — engine-truth numbers, not wall-clock averages.

Why: llama-benchy's "t/s (total)" at concurrency c averages over a window that includes the
staggered prefill ramp-up and the ramp-down (streams finish at different times), so at c4/c8 with
tg512 most of the window has < c streams decoding — the number swings +-30% between runs of the
same stack, while its "peak" column (instantaneous, all streams live) is stable. This probe samples
vLLM's own Prometheus counters every `--dt` seconds while `c` long generations run and reports
throughput over the samples where num_requests_running == c only. Spec-decode acceptance from the
same counters. Short prompts + min_tokens so every stream decodes for the whole window.

usage: decode_ss.py --url http://localhost:8020 --model qwen3.8-27b --conc 1 2 4 8 --tokens 1024 --runs 3
"""
import argparse, json, statistics, threading, time, urllib.request, random

def metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
        txt = r.read().decode()
    out = {}
    for line in txt.splitlines():
        if line.startswith("vllm:generation_tokens_total") or line.startswith("vllm:num_requests_running") \
           or line.startswith("vllm:spec_decode_num_accepted_tokens_total") or line.startswith("vllm:spec_decode_num_draft_tokens_total") \
           or line.startswith("vllm:prompt_tokens_total"):
            k = line.split("{")[0].split(" ")[0]; v = float(line.rsplit(" ", 1)[1]); out[k] = v
    return out

def stream(base, model, prompt, tokens, results, idx):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens, "min_tokens": tokens,
            "temperature": 0.6, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; last = None; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:]); last = time.time()
            if first is None and d.get("choices") and (d["choices"][0].get("delta") or {}).get("content") is not None: first = last
            if d.get("usage"): usage = d["usage"]
    results[idx] = {"ttft": (first or last) - t0, "end": last, "start": t0, "first": first, "usage": usage}

FILLER = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra "
          "tango uniform victor whiskey xray yankee zulu river mountain harbor lantern copper meadow orbit velvet canyon ember ").split()
CODE_TASKS = ["a Python library for parsing and evaluating arithmetic expressions with a tokenizer, a Pratt parser and an evaluator, with unit tests",
              "a TypeScript LRU cache with TTL, generics, and a full test suite using vitest", "a Go HTTP server with middleware, JSON routes, graceful shutdown and table-driven tests",
              "a Rust command-line tool that tails a log file and aggregates per-minute error counts, with clap and tests",
              "a Python asyncio job queue with retries, backoff, and a SQLite-backed store, with pytest tests"]

def run(base, model, c, tokens, dt, seed, kind="prose", ctx=0):
    rng = random.Random(seed)
    topics = ["the history of the Rhône river", "how a refrigerator works", "the rules of go", "a tour of Lisbon", "why the sky is blue",
              "sourdough baking", "the Apollo program", "how TCP congestion control works", "the water cycle", "a day in a medieval town"]
    if kind == "code":
        prompts = [f"Implement {rng.choice(CODE_TASKS)}. Output complete, well-commented source files; keep going until everything is implemented." for _ in range(c)]
    else:
        prompts = [f"Write a long, detailed essay about {rng.choice(topics)}. Do not stop early; keep elaborating with new sections." for _ in range(c)]
    if ctx:
        prompts = [" ".join(rng.choice(FILLER) for _ in range(int(ctx / 1.3))) + "\n\nIgnore the words above. " + p for p in prompts]
    samples = []; stop = threading.Event()
    def sampler():
        while not stop.is_set():
            try: samples.append((time.time(), metrics(base)))
            except Exception: pass
            time.sleep(dt)
    th = threading.Thread(target=sampler, daemon=True); th.start()
    res = [None] * c
    ths = [threading.Thread(target=stream, args=(base, model, prompts[i], tokens, res, i)) for i in range(c)]
    for t in ths: t.start()
    for t in ths: t.join()
    stop.set(); th.join()
    # steady-state window: samples where num_running == c (and after every stream has its first token)
    first_all = max(r["first"] or r["end"] for r in res); end_min = min(r["end"] for r in res)
    ss = [(t, m) for t, m in samples if m.get("vllm:num_requests_running", -1) == c and first_all <= t <= end_min]
    if len(ss) < 2:
        return {"c": c, "error": f"no steady-state window (samples={len(samples)}, ss={len(ss)})"}
    (t0, m0), (t1, m1) = ss[0], ss[-1]
    gen = m1["vllm:generation_tokens_total"] - m0["vllm:generation_tokens_total"]; T = t1 - t0
    acc = m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m0.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    drf = m1.get("vllm:spec_decode_num_draft_tokens_total", 0) - m0.get("vllm:spec_decode_num_draft_tokens_total", 0)
    comp = [r["usage"]["completion_tokens"] for r in res if r["usage"]]
    per_stream_e2e = [r["usage"]["completion_tokens"] / (r["end"] - r["first"]) for r in res if r["usage"] and r["first"]]
    return {"c": c, "ss_window_s": round(T, 1), "ss_agg_tps": round(gen / T, 1), "ss_per_stream_tps": round(gen / T / c, 1),
            "accept_per_draft": round(acc / drf, 3) if drf else None, "e2e_per_stream_tps_median": round(statistics.median(per_stream_e2e), 1),
            "ttft_s_median": round(statistics.median(r["ttft"] for r in res), 2), "completion_tokens": comp}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8020"); ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--conc", type=int, nargs="+", default=[1, 2, 4, 8]); ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=3); ap.add_argument("--dt", type=float, default=0.5); ap.add_argument("--out", default=None)
    ap.add_argument("--kind", choices=["prose", "code"], default="prose"); ap.add_argument("--ctx", type=int, default=0, help="filler tokens prepended (deep-context decode)")
    a = ap.parse_args()
    for c in a.conc:
        rs = [run(a.url, a.model, c, a.tokens, a.dt, seed=f"{c}-{i}", kind=a.kind, ctx=a.ctx) for i in range(a.runs)]
        good = [r for r in rs if "error" not in r]
        if not good:
            print(f"c{c}: {rs}"); continue
        agg = [r["ss_agg_tps"] for r in good]; acc = [r["accept_per_draft"] for r in good if r["accept_per_draft"] is not None]
        summ = {"c": c, "kind": a.kind, "ctx": a.ctx, "runs": len(good), "ss_agg_tps_median": statistics.median(agg), "ss_agg_tps_min_max": [min(agg), max(agg)],
                "ss_per_stream_tps_median": round(statistics.median(agg) / c, 1), "accept_per_draft_median": statistics.median(acc) if acc else None,
                "ttft_s_median": statistics.median(r["ttft_s_median"] for r in good), "ss_window_s": [r["ss_window_s"] for r in good]}
        print("RESULT", json.dumps(summ), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps({"summary": summ, "runs": rs}) + "\n")

if __name__ == "__main__":
    main()
