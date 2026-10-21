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

--client (R808, TabbyAPI :8022, which exports no running-request gauge and books generated tokens at completion only):
the steady state is measured client-side instead. Each stream's decode rate = (completion_tokens - 1) / (last - first
streamed delta, reasoning included); ss_agg_tps = the sum over the c streams (all start together with min_tokens =
max_tokens, loop_detect_window 0, thinking off as peer_gap's peers); ss_window_s = the span in which all c streams were streaming (min end - max first),
reported beside it. --temperature (default 0.6 = every historical invocation) sets the sampling temperature; 0 = greedy.
"""
import argparse, json, statistics, threading, time, urllib.request, random

# Power collection is default-on; a missing helper cannot interrupt the probe.
import os as _power_os, sys as _power_sys
_power_sys.path.insert(0, _power_os.environ.get("FLAN_POWER_LIB", "/srv/qwen5090/lib"))
try:
    from flan_power import Phase as _PowerPhase, result_dir as _power_dir
except Exception as _power_error:
    print(f"flan_power unavailable: {_power_error}", file=_power_sys.stderr)
    import contextlib as _power_context
    class _PowerPhase(_power_context.nullcontext):
        def __init__(self, *args, **kwargs): super().__init__(self)
        def window(self, *args, **kwargs): pass
    def _power_dir(out=None): return _power_os.environ.get("FLAN_POWER_DIR", ".")

def metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
        txt = r.read().decode()
    out = {}
    # sglang exports the same counters under its own names (needs --enable-metrics); normalize to the vllm keys
    ALIAS = {"sglang:generation_tokens_total": "vllm:generation_tokens_total",
             "sglang:num_running_reqs": "vllm:num_requests_running",
             "sglang:prompt_tokens_total": "vllm:prompt_tokens_total",
             "sglang:spec_accept_length": "sglang:spec_accept_length",
             "sglang:gen_throughput": "sglang:gen_throughput",
             "sglang:spec_accept_rate": "sglang:spec_accept_rate"}
    KEYS = ("vllm:generation_tokens_total", "vllm:num_requests_running", "vllm:spec_decode_num_accepted_tokens_total",
            "vllm:spec_decode_num_draft_tokens_total", "vllm:prompt_tokens_total") + tuple(ALIAS)
    for line in txt.splitlines():
        if line.startswith(KEYS):
            k = line.split("{")[0].split(" ")[0]; v = float(line.rsplit(" ", 1)[1]); out[ALIAS.get(k, k)] = v
    return out

def stream(base, model, prompt, tokens, results, idx, temperature=0.6, client=False):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens, "min_tokens": tokens,
            "temperature": temperature, "stream": True, "stream_options": {"include_usage": True}}
    if client:
        body["loop_detect_window"] = 0   # TabbyAPI (loop-think): never stop a forced-length stream early
        body["chat_template_kwargs"] = {"enable_thinking": False}   # the peer_gap peers' regime
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; last = None; last_text = None; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:]); last = time.time()
            dl = (d["choices"][0].get("delta") or {}) if d.get("choices") else {}
            got = (dl.get("content") or dl.get("reasoning_content")) if client else dl.get("content") is not None
            if got: last_text = last
            if first is None and got: first = last
            if d.get("usage"): usage = d["usage"]
    results[idx] = {"ttft": (first or last) - t0, "end": last, "last_text": last_text, "start": t0, "first": first, "usage": usage}

FILLER = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra "
          "tango uniform victor whiskey xray yankee zulu river mountain harbor lantern copper meadow orbit velvet canyon ember ").split()
CODE_TASKS = ["a Python library for parsing and evaluating arithmetic expressions with a tokenizer, a Pratt parser and an evaluator, with unit tests",
              "a TypeScript LRU cache with TTL, generics, and a full test suite using vitest", "a Go HTTP server with middleware, JSON routes, graceful shutdown and table-driven tests",
              "a Rust command-line tool that tails a log file and aggregates per-minute error counts, with clap and tests",
              "a Python asyncio job queue with retries, backoff, and a SQLite-backed store, with pytest tests"]

def _run(base, model, c, tokens, dt, seed, kind="prose", ctx=0, temperature=0.6, client=False):
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
    th = threading.Thread(target=sampler, daemon=True)
    if not client: th.start()   # client mode: no /metrics scrape during the measurement
    res = [None] * c
    ths = [threading.Thread(target=stream, args=(base, model, prompts[i], tokens, res, i, temperature, client)) for i in range(c)]
    for t in ths: t.start()
    for t in ths: t.join()
    stop.set()
    if not client: th.join()
    failed = sum(r is None for r in res)
    if failed:  # a stream thread died (engine error mid-request): report instead of a TypeError on the None entry (R190 2026-09-05)
        return {"c": c, "error": f"{failed}/{c} streams failed (engine error?)"}
    if client:
        rates = [(r["usage"]["completion_tokens"] - 1) / (r["end"] - r["first"]) for r in res
                 if r["usage"] and r["first"] and r["end"] > r["first"]]
        if len(rates) < c:
            return {"c": c, "error": f"client mode: {c - len(rates)}/{c} streams without usage or first token"}
        first_all = max(r["first"] for r in res); end_min = min(r["end"] for r in res)
        return {"_power_window": dict(t0=first_all, t1=min(r["last_text"] or r["end"] for r in res), tokens_out=sum(rates)*(min(r["last_text"] or r["end"] for r in res)-first_all), notes=["Historical client rate includes last SSE frame; power ends at last text and uses that same rate × shorter overlap."]), "c": c, "mode": "client", "ss_window_s": round(end_min - first_all, 1), "ss_agg_tps": round(sum(rates), 1),
                "ss_per_stream_tps": round(sum(rates) / c, 1), "accept_per_draft": None,
                "e2e_per_stream_tps_median": round(statistics.median(rates), 1),
                "ttft_s_median": round(statistics.median(r["first"] - r["start"] for r in res), 2),
                "completion_tokens": [r["usage"]["completion_tokens"] for r in res]}
    # steady-state window: samples where num_running == c (and after every stream has its first token)
    first_all = max(r["first"] or r["end"] for r in res); end_min = min(r["end"] for r in res)
    ss = [(t, m) for t, m in samples if m.get("vllm:num_requests_running", -1) == c and first_all <= t <= end_min]
    if len(ss) < 2:
        return {"c": c, "error": f"no steady-state window (samples={len(samples)}, ss={len(ss)})"}
    (t0, m0), (t1, m1) = ss[0], ss[-1]
    T = t1 - t0
    if "sglang:gen_throughput" in m1:  # sglang: token counters only tick at request completion; use the live gauge
        tps_gauge = [m["sglang:gen_throughput"] for _, m in ss if m.get("sglang:gen_throughput", 0) > 0]
        if not tps_gauge:
            return {"c": c, "error": f"sglang gen_throughput gauge empty over ss window ({len(ss)} samples)"}
        rate = statistics.median(tps_gauge)
        acc_rates = [m["sglang:spec_accept_rate"] for _, m in ss if m.get("sglang:spec_accept_rate", 0) > 0]
        comp = [r["usage"]["completion_tokens"] for r in res if r["usage"]]
        per_stream_e2e = [r["usage"]["completion_tokens"] / (r["end"] - r["first"]) for r in res if r["usage"] and r["first"]]
        return {"_power_window": dict(t0=t0, t1=t1, tokens_out=rate*T, notes=["SGlang live throughput gauge estimator × steady window."]), "c": c, "ss_window_s": round(T, 1), "ss_agg_tps": round(rate, 1), "ss_per_stream_tps": round(rate / c, 1),
                "accept_per_draft": round(statistics.median(acc_rates), 3) if acc_rates else None,
                "e2e_per_stream_tps_median": round(statistics.median(per_stream_e2e), 1) if per_stream_e2e else None,
                "ttft_s_median": round(statistics.median(r["ttft"] for r in res), 2), "completion_tokens": comp}
    gen = m1["vllm:generation_tokens_total"] - m0["vllm:generation_tokens_total"]
    acc = m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m0.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    drf = m1.get("vllm:spec_decode_num_draft_tokens_total", 0) - m0.get("vllm:spec_decode_num_draft_tokens_total", 0)
    comp = [r["usage"]["completion_tokens"] for r in res if r["usage"]]
    per_stream_e2e = [r["usage"]["completion_tokens"] / (r["end"] - r["first"]) for r in res if r["usage"] and r["first"]]
    return {"_power_window": dict(t0=t0, t1=t1, tokens_out=gen), "c": c, "ss_window_s": round(T, 1), "ss_agg_tps": round(gen / T, 1), "ss_per_stream_tps": round(gen / T / c, 1),
            "accept_per_draft": round(acc / drf, 3) if drf else None, "e2e_per_stream_tps_median": round(statistics.median(per_stream_e2e), 1),
            "ttft_s_median": round(statistics.median(r["ttft"] for r in res), 2), "completion_tokens": comp}

def run(base, model, c, tokens, dt, seed, kind="prose", ctx=0, temperature=0.6, client=False, power_out=None):
    with _PowerPhase(_power_dir(power_out), f"decode-{kind}-c{c}", c, "decode") as p:
        result = _run(base, model, c, tokens, dt, seed, kind, ctx, temperature, client)
        window = result.pop("_power_window", None)
        if window: p.window(**window)
        return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8020"); ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--conc", type=int, nargs="+", default=[1, 2, 4, 8]); ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=3); ap.add_argument("--dt", type=float, default=0.5); ap.add_argument("--out", default=None)
    ap.add_argument("--kind", choices=["prose", "code"], default="prose"); ap.add_argument("--ctx", type=int, default=0, help="filler tokens prepended (deep-context decode)")
    ap.add_argument("--seed-prefix", default="", help="prepended to the per-run prompt seed. Default '' keeps every historical invocation byte-identical. "
                    "Set it per ARM when arms must not share prompts: identical prompts hit the prefix cache, which collapses TTFT on the second arm "
                    "and voids any prefill comparison (R208 2026-09-07: 100K TTFT 16.58 s on the first arm, 0.93 s on the second). Decode is unaffected.")
    ap.add_argument("--temperature", type=float, default=0.6, help="0.6 = every historical invocation; 0 = greedy (R808)")
    ap.add_argument("--client", action="store_true", help="client-side steady state (TabbyAPI: no running-request gauge)")
    a = ap.parse_args()
    for c in a.conc:
        rs = [run(a.url, a.model, c, a.tokens, a.dt, seed=f"{a.seed_prefix}{c}-{i}", kind=a.kind, ctx=a.ctx,
                  temperature=a.temperature, client=a.client, power_out=a.out) for i in range(a.runs)]
        good = [r for r in rs if "error" not in r]
        if not good:
            print(f"c{c}: {rs}"); continue
        agg = [r["ss_agg_tps"] for r in good]; acc = [r["accept_per_draft"] for r in good if r["accept_per_draft"] is not None]
        summ = {"c": c, **({"mode": "client"} if a.client else {}), "kind": a.kind, "ctx": a.ctx, "runs": len(good), "ss_agg_tps_median": statistics.median(agg), "ss_agg_tps_min_max": [min(agg), max(agg)],
                "ss_per_stream_tps_median": round(statistics.median(agg) / c, 1), "accept_per_draft_median": statistics.median(acc) if acc else None,
                "ttft_s_median": statistics.median(r["ttft_s_median"] for r in good), "ss_window_s": [r["ss_window_s"] for r in good]}
        print("RESULT", json.dumps(summ), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps({"summary": summ, "runs": rs}) + "\n")

if __name__ == "__main__":
    main()
