#!/usr/bin/env python3
"""Admission probe (R177/R178): fire N concurrent chat streams and sample vllm:num_requests_running / _waiting every 0.25 s.
Prints the max running seen and the (running, waiting) histogram. R177: the daily at --max-num-seqs 16 held (15, 1) under 16 streams."""
import argparse, collections, json, threading, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8029"); ap.add_argument("--model", default="qwen3.8-27b")
ap.add_argument("--conc", type=int, required=True); ap.add_argument("--tokens", type=int, default=500); a = ap.parse_args()
def stream():
    body = json.dumps({"model": a.model, "messages": [{"role": "user", "content": "Write a long Python module implementing a red-black tree with insert, delete, and iteration, with docstrings."}],
                       "max_tokens": a.tokens, "temperature": 0.6, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    try: urllib.request.urlopen(urllib.request.Request(a.url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}), timeout=600).read()
    except Exception as e: print("stream error:", e)
ts = [threading.Thread(target=stream) for _ in range(a.conc)]; [t.start() for t in ts]
cnt = collections.Counter(); t0 = time.time(); mx = 0
while any(t.is_alive() for t in ts):
    try:
        m = urllib.request.urlopen(a.url + "/metrics", timeout=5).read().decode(); r = w = None
        for line in m.splitlines():
            if line.startswith("vllm:num_requests_running{"): r = int(float(line.split()[-1]))
            if line.startswith("vllm:num_requests_waiting{"): w = int(float(line.split()[-1]))
        cnt[(r, w)] += 1; mx = max(mx, r or 0)
    except Exception: pass
    time.sleep(0.25)
print(json.dumps({"conc": a.conc, "max_running": mx, "elapsed_s": round(time.time() - t0, 1), "hist": [[k[0], k[1], v] for k, v in sorted(cnt.items(), key=lambda x: -x[1])[:6]]}))
