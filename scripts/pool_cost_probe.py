#!/usr/bin/env python3
"""Per-request fixed KV-pool cost (R200, 2026-09-05). Fires c identical short-prompt generations, samples /metrics every 0.5 s,
and reports kv_cache_usage_perc at the samples where every request is running: the pool a request occupies just by existing
(GDN/mamba state slots + its first attention block). On the R197 daily (nvfp4 KV, DFlash2 ns7, pool 1,052,277) this is 2.72 % of
the pool = 28,613 tokens-equivalent per request, constant over 1,200 generated tokens, which is why a SEQS 64 boot ran 36 requests
and queued the other 28 (r200-c32c64). Usage: pool_cost_probe.py --url U [--conc 8 14] [--tokens 1200] [--pool 1052277]"""
import argparse, json, threading, time, urllib.request

def metrics(u):
    t = urllib.request.urlopen(u + "/metrics", timeout=5).read().decode(); d = {}
    for l in t.splitlines():
        for k in ("vllm:num_requests_running", "vllm:kv_cache_usage_perc", "vllm:num_requests_waiting"):
            if l.startswith(k + "{") or l.startswith(k + " "): d[k] = float(l.rsplit(" ", 1)[1])
    return d

def one(u, model, i, n):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": f"Write a long essay number {i} about rivers. Keep going with new sections."}],
                       "max_tokens": n, "min_tokens": n, "temperature": 0.6}).encode()
    try: urllib.request.urlopen(urllib.request.Request(u + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}), timeout=600).read()
    except Exception as e: print("req err", e)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8020"); ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--conc", type=int, nargs="+", default=[8]); ap.add_argument("--tokens", type=int, default=1200); ap.add_argument("--pool", type=int, default=0)
    a = ap.parse_args()
    pool = a.pool
    if not pool:
        t = urllib.request.urlopen(a.url + "/metrics", timeout=5).read().decode()
        for l in t.splitlines():
            if l.startswith("vllm:cache_config_info") and 'kv_cache_size_tokens="' in l: pool = int(l.split('kv_cache_size_tokens="')[1].split('"')[0])
    for c in a.conc:
        base = metrics(a.url)
        ths = [threading.Thread(target=one, args=(a.url, a.model, i, a.tokens)) for i in range(c)]; [t.start() for t in ths]
        samples = []; t0 = time.time()
        while any(t.is_alive() for t in ths) and time.time() - t0 < 900:
            try: samples.append((time.time() - t0, metrics(a.url)))
            except Exception: pass
            time.sleep(0.5)
        [t.join() for t in ths]
        full = [(t, d) for t, d in samples if d.get("vllm:num_requests_running", 0) == c]
        maxrun = max((d.get("vllm:num_requests_running", 0) for _, d in samples), default=0); maxwait = max((d.get("vllm:num_requests_waiting", 0) for _, d in samples), default=0)
        if full:
            (ta, da), (tb, db) = full[0], full[-1]
            ua = da["vllm:kv_cache_usage_perc"] - base.get("vllm:kv_cache_usage_perc", 0); ub = db["vllm:kv_cache_usage_perc"] - base.get("vllm:kv_cache_usage_perc", 0)
            print(json.dumps({"c": c, "pool_tokens": pool, "samples": len(samples), "all_running_samples": len(full), "usage_pct_first": round(ua * 100, 2), "usage_pct_last": round(ub * 100, 2),
                              "tok_per_req_first": round(ua / c * pool), "tok_per_req_last": round(ub / c * pool), "pct_per_req": round(ub / c * 100, 3),
                              "max_concurrent_by_pool": int(100 / (ub / c * 100)) if ub else None, "t_first_s": round(ta, 1), "t_last_s": round(tb, 1), "max_running": maxrun, "max_waiting": maxwait}))
        else:
            print(json.dumps({"c": c, "error": "never all running", "max_running": maxrun, "max_waiting": maxwait, "samples": len(samples)}))
        time.sleep(3)

if __name__ == "__main__":
    main()
