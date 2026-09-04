#!/usr/bin/env python3
"""KV capacity probe (R178b): send `conc` concurrent requests of `ctx` filler tokens, sample vllm:kv_cache_usage_perc,
num_requests_running and num_preemptions_total; print the maxima and each request's latency and prompt_tokens.
R178: on the served route a request costs about 6.4% of the pool at admission plus about 0.155% per 1K context tokens."""
import argparse, json, random, threading, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8020"); ap.add_argument("--model", default="qwen3.8-27b")
ap.add_argument("--ctx", type=int, required=True); ap.add_argument("--conc", type=int, required=True); ap.add_argument("--tokens", type=int, default=32)
ap.add_argument("--seed", type=int, default=int(time.time()) % 100000, help="distinct per run, or the prefix cache serves the repeat")
ap.add_argument("--ignore-eos", action="store_true", help="hold the request alive for max_tokens so co-residency can be observed")
a = ap.parse_args()
def met(name):
    for line in urllib.request.urlopen(a.url + "/metrics", timeout=5).read().decode().splitlines():
        if line.startswith(name + "{"): return float(line.split()[-1])
    return 0.0
words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lam mu".split()
p0 = met("vllm:num_preemptions_total"); res = [None] * a.conc
def one(i):
    rng = random.Random(a.seed * 1000 + i)
    filler = " ".join(rng.choice(words) for _ in range(int(a.ctx / 1.3))) + "\n\nIgnore the words above. " if a.ctx else ""
    body = json.dumps({"model": a.model, "messages": [{"role": "user", "content": filler + "Say hello."}], "max_tokens": a.tokens, "temperature": 0.0, "ignore_eos": a.ignore_eos,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t0 = time.time()
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request(a.url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}), timeout=3600).read())
        res[i] = [round(time.time() - t0, 1), r["usage"]["prompt_tokens"]]
    except Exception as e: res[i] = ["ERR " + str(e)[:80]]
ts = [threading.Thread(target=one, args=(i,)) for i in range(a.conc)]; [t.start() for t in ts]
mx = mxrun = 0.0
while any(t.is_alive() for t in ts):
    try: mx = max(mx, met("vllm:kv_cache_usage_perc")); mxrun = max(mxrun, met("vllm:num_requests_running"))
    except Exception: pass
    time.sleep(0.5)
print(json.dumps({"ctx": a.ctx, "conc": a.conc, "seed": a.seed, "max_kv_usage_pct": round(mx * 100, 1), "max_running": int(mxrun), "preemptions": int(met("vllm:num_preemptions_total") - p0), "requests": res}))
