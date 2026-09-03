#!/usr/bin/env python3
"""Warm-revisit probe (R147): send an identical long prompt twice, report TTFT1/TTFT2
and the prefix-cache hit/query deltas around each send. Proof of the pr53479 fix:
send 2 cached-token delta rises by ~one align block (2864) vs the unpatched baseline.
Usage: warm-revisit.py --url http://127.0.0.1:8029 --model qwen3.8-27b --ctx 32000 [--salt X]"""
import argparse, json, time, urllib.request, uuid, re

def metrics(url):
    t = urllib.request.urlopen(url + "/metrics", timeout=10).read().decode()
    out = {}
    # GPU prefix cache; external (KV-connector) prefix cache; the tiering connector's fs-tier read/hit counters
    # (R166: a tier hit across a restart does not show in prefix_cache_hits — it is external_prefix_cache_hits)
    for name in ("prefix_cache_hits", "prefix_cache_queries", "external_prefix_cache_hits", "external_prefix_cache_queries",
                 "kv_offload_tiering_read_bytes", "kv_offload_tiering_chunk_hits", "kv_offload_tiering_chunk_queries"):
        s = sum(float(m.group(1)) for m in re.finditer(rf"vllm:{name}(?:_total)?(?:{{[^}}]*}})? ([0-9.e+]+)", t))
        out[name] = s
    return out

def send(url, model, prompt):
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": 8, "temperature": 0.0, "stream": True}).encode()
    req = urllib.request.Request(url + "/v1/completions", body, {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.strip().startswith(b"data:") and b"[DONE]" not in line:
                ttft = ttft or time.time() - t0
    return ttft, time.time() - t0

p = argparse.ArgumentParser()
p.add_argument("--url", required=True); p.add_argument("--model", required=True)
p.add_argument("--ctx", type=int, default=32000); p.add_argument("--salt", default=None)
p.add_argument("--flood", type=int, default=0, help="between sends: N fresh prompts to evict GPU blocks")
p.add_argument("--flood-ctx", type=int, default=90000)
a = p.parse_args()
salt = a.salt or uuid.uuid4().hex
words = ("The archive room catalogues item %d with reference code %s under revision seven. " % (0, salt))
prompt = f"Nonce: {salt}.\n" + " ".join(
    f"The archive room catalogues item {i} with reference code A{i*7%9973} under revision {i%9}." for i in range(a.ctx // 14)
) + "\nSummarize the catalogue in one word:"
res = {}
for n in (1, 2):
    m0 = metrics(a.url)
    ttft, total = send(a.url, a.model, prompt)
    time.sleep(2)
    m1 = metrics(a.url)
    res[f"send{n}"] = {"ttft_s": round(ttft, 3),
                       "hits_delta": m1["prefix_cache_hits"] - m0["prefix_cache_hits"],
                       "queries_delta": m1["prefix_cache_queries"] - m0["prefix_cache_queries"],
                       "ext_hits_delta": m1["external_prefix_cache_hits"] - m0["external_prefix_cache_hits"],
                       "ext_queries_delta": m1["external_prefix_cache_queries"] - m0["external_prefix_cache_queries"],
                       "tier_chunk_hits_delta": m1["kv_offload_tiering_chunk_hits"] - m0["kv_offload_tiering_chunk_hits"],
                       "tier_chunk_queries_delta": m1["kv_offload_tiering_chunk_queries"] - m0["kv_offload_tiering_chunk_queries"],
                       "tier_read_mb_delta": round((m1["kv_offload_tiering_read_bytes"] - m0["kv_offload_tiering_read_bytes"]) / 1048576, 1)}
    if n == 1:
        time.sleep(8)  # let async stores settle
        for f in range(a.flood):  # evict: fresh unique prompts fill the GPU pool
            fp = f"Flood {uuid.uuid4().hex}.\n" + " ".join(
                f"Filler item {i} code B{i*13%9973} rev {i%7}." for i in range(a.flood_ctx // 9)
            ) + "\nOne word:"
            try: send(a.url, a.model, fp)
            except Exception as e: print(f"flood {f} failed: {e}")
        time.sleep(5)
res["salt"] = salt; res["ctx"] = a.ctx
print("RESULT " + json.dumps(res))
