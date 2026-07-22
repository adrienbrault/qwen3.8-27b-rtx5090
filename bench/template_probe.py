"""Behavioural probe for a Qwen chat template served by vLLM.

Checks the things a speed benchmark can't: does the model emit parseable tool calls,
does a chat->tool->chat loop survive (the "agentic abort" failure), and does a plain
chat turn after tools still work.

Every check runs independently — an early failure is reported, not crashed on.
Exit code: 0 if all checks pass, 1 otherwise. --json for machine-readable output.

usage: python3 template_probe.py <base_url> [label] [--json]
"""

import json
import sys
import urllib.request

args = [a for a in sys.argv[1:] if a != "--json"]
AS_JSON = "--json" in sys.argv
BASE = args[0].rstrip("/")
LABEL = args[1] if len(args) > 1 else BASE
MODEL = "qwen3.6-27b"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time in a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]


def post(payload, timeout=180):
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


results = {}
notes = {}


def check(name, fn):
    """Run one check in isolation; a raised exception is a FAIL with the error noted."""
    try:
        ok, note = fn()
        results[name] = bool(ok)
        if note is not None:
            notes[name] = note
    except Exception as e:  # noqa: BLE001 — a probe must report, never crash
        results[name] = False
        notes[name] = f"EXCEPTION: {type(e).__name__}: {e}"


# 1. single tool call — its output feeds check 3/4 if it succeeds
first_calls = []


def single_tool_call():
    global first_calls
    r = post({
        "model": MODEL,
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "tools": TOOLS,
        "max_tokens": 1500,
    })
    msg = r["choices"][0]["message"]
    first_calls = msg.get("tool_calls") or []
    ok = len(first_calls) == 1 and first_calls[0]["function"]["name"] == "get_weather"
    return ok, None


check("single_tool_call", single_tool_call)

# 2. parallel tool calls (the multi-call drop bug)


def parallel_tool_calls():
    r = post({
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": "What's the weather AND the current time in Tokyo? Call both tools.",
        }],
        "tools": TOOLS,
        "max_tokens": 1500,
    })
    calls = r["choices"][0]["message"].get("tool_calls") or []
    names = sorted(c["function"]["name"] for c in calls)
    return names == ["get_time", "get_weather"], f"got {names}"


check("parallel_tool_calls", parallel_tool_calls)

# 3 + 4. chat -> tool result -> chat continuation, then plain chat after tools.
# Uses check 1's call when available, else a synthetic call — so these checks still
# run (and report) when check 1 failed, instead of dying with a NameError.
convo = [
    {"role": "user", "content": "What's the weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": (first_calls[0].get("id", "call_1") if first_calls else "call_1"),
            "type": "function",
            "function": {
                "name": (first_calls[0]["function"]["name"] if first_calls else "get_weather"),
                "arguments": (first_calls[0]["function"]["arguments"] if first_calls else '{"city": "Paris"}'),
            },
        }],
    },
    {
        "role": "tool",
        "tool_call_id": (first_calls[0].get("id", "call_1") if first_calls else "call_1"),
        "content": '{"temp_c": 18, "condition": "rain"}',
    },
]


def tool_result_continuation():
    r = post({"model": MODEL, "messages": convo, "tools": TOOLS, "max_tokens": 1500})
    final = (r["choices"][0]["message"].get("content") or "").strip()
    convo.append({"role": "assistant", "content": final or "ok"})
    return ("18" in final) and len(final) > 10, final[:100]


check("tool_result_continuation", tool_result_continuation)


def chat_after_tools():
    if len(convo) == 3:  # continuation check died before appending — keep the convo legal
        convo.append({"role": "assistant", "content": "ok"})
    convo.append({"role": "user", "content": "Thanks. Now, in one word: is that good weather?"})
    r = post({"model": MODEL, "messages": convo, "tools": TOOLS, "max_tokens": 1500})
    followup = (r["choices"][0]["message"].get("content") or "").strip()
    return len(followup) > 0, followup[:60]


check("chat_after_tools", chat_after_tools)

score = sum(results.values())
total = len(results)
if AS_JSON:
    print(json.dumps({"label": LABEL, "pass": score == total, "score": f"{score}/{total}",
                      "results": results, "notes": notes}))
else:
    print(f"\n=== {LABEL}")
    for k, v in results.items():
        note = f"   ({notes[k]!r})" if k in notes else ""
        print(f"  {'PASS' if v else 'FAIL'}  {k}{note}")
    print(f"  -> {score}/{total}")
sys.exit(0 if score == total else 1)
