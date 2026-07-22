"""Prefix-cache probe: measure it by TIMING, judge it by timing, report the counter as color.

vLLM's `prefix_cache_hits_total` reports 0% on this hybrid model even when the cache
works (README gotcha #5), so a counter can never be the verdict. This probe drives a
realistic agentic pattern (same long system prompt + growing turn history), times each
turn end-to-end, and passes/fails on the measured speedup of cached turns over the
cold first turn.

PASS: mean e2e latency of turns 2..N is at least SPEEDUP_MIN times faster per prompt
token than turn 1 (prefill dominates e2e at this prompt size, so per-token normalized
latency isolates the cache effect). Exit code 0/1. --json for machine output.

usage: python3 prefix_probe.py <base_url> [label] [--json]
env:   SPEEDUP_MIN (default 2.0)
"""

import json
import os
import re
import sys
import time
import urllib.request

args = [a for a in sys.argv[1:] if a != "--json"]
AS_JSON = "--json" in sys.argv
BASE = args[0].rstrip("/")
LABEL = args[1] if len(args) > 1 else BASE
MODEL = "qwen3.6-27b"
SPEEDUP_MIN = float(os.environ.get("SPEEDUP_MIN", "2.0"))

SYSTEM = (
    "You are a coding assistant working in a large repository. "
    "Follow the house style. Be concise. " + ("Context filler. " * 400)
)


def post(messages):
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        json.dumps({"model": MODEL, "messages": messages, "max_tokens": 30}).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    r = json.load(urllib.request.urlopen(req, timeout=180))
    elapsed = time.monotonic() - t0
    return r["choices"][0]["message"].get("content") or "", r["usage"], elapsed


def counter():
    try:
        with urllib.request.urlopen(f"{BASE}/metrics", timeout=30) as f:
            body = f.read().decode()
        q = re.search(r"^vllm:prefix_cache_queries_total\S*\s+([\d.e+]+)$", body, re.M)
        h = re.search(r"^vllm:prefix_cache_hits_total\S*\s+([\d.e+]+)$", body, re.M)
        return (float(q.group(1)) if q else 0.0, float(h.group(1)) if h else 0.0)
    except Exception:  # metrics endpoint optional — the verdict never depends on it
        return (0.0, 0.0)


q0, h0 = counter()

convo = [{"role": "system", "content": SYSTEM}]
questions = [
    "What is 2+2? One word.",
    "And 3+3? One word.",
    "And 4+4? One word.",
    "And 5+5? One word.",
    "And 6+6? One word.",
]
turns = []
for question in questions:
    convo.append({"role": "user", "content": question})
    reply, usage, elapsed = post(convo)
    convo.append({"role": "assistant", "content": reply})
    turns.append({"e2e_s": round(elapsed, 3), "prompt_tokens": usage["prompt_tokens"]})

q1, h1 = counter()
dq, dh = q1 - q0, h1 - h0

# per-prompt-token latency: turn 1 pays the full prefill; later turns should mostly hit
cold = turns[0]["e2e_s"] / turns[0]["prompt_tokens"]
warm = [t["e2e_s"] / t["prompt_tokens"] for t in turns[1:]]
warm_mean = sum(warm) / len(warm)
speedup = cold / warm_mean if warm_mean > 0 else 0.0
ok = speedup >= SPEEDUP_MIN

out = {
    "label": LABEL,
    "pass": ok,
    "speedup_per_prompt_token": round(speedup, 2),
    "threshold": SPEEDUP_MIN,
    "cold_turn_s": turns[0]["e2e_s"],
    "warm_turns_s": [t["e2e_s"] for t in turns[1:]],
    "turns": turns,
    "counter_info_only": f"{dh:.0f}/{dq:.0f} blocks ({(dh / dq * 100) if dq else 0:.1f}%) — known-unreliable on this hybrid",
}
if AS_JSON:
    print(json.dumps(out))
else:
    print(f"=== {LABEL}")
    print(f"  cold turn: {out['cold_turn_s']}s @ {turns[0]['prompt_tokens']} prompt tok; warm turns: {out['warm_turns_s']}")
    print(f"  per-prompt-token speedup: {out['speedup_per_prompt_token']}x (threshold {SPEEDUP_MIN}x) -> {'PASS' if ok else 'FAIL'}")
    print(f"  native counter (info only, unreliable here): {out['counter_info_only']}")
sys.exit(0 if ok else 1)
