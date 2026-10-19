#!/usr/bin/env python3
"""Tool-call smoke for the BA4 lean gate: a handful of chat requests that should each end in one tool call.

  toolsmoke.py <base_url> <out.json>

A request passes when the API returns at least one tool call whose name is in the offered tools and whose arguments
parse as a JSON object. Greedy (temperature 0), so the same engine answers the same way; the gate compares arms.
"""
import json, sys, urllib.request

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file from the repository.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write a file in the repository.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_command", "description": "Run a shell command and return its output.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]
PROMPTS = [
    "Show me what is in src/main.py.",
    "Run the test suite with pytest -q and tell me what fails.",
    "Create a file hello.py that prints 'hello world'.",
    "Write a Python function fib(n) with a docstring into fib.py.",
    "How many lines does README.md have? Use a shell command.",
]


def main(url, out):
    rows = []
    names = {t["function"]["name"] for t in TOOLS}
    for p in PROMPTS:
        body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": p}], "tools": TOOLS,
                "tool_choice": "auto", "temperature": 0, "max_tokens": 8192}
        req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        msg = json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        ok, why = bool(calls), "no tool call"
        for c in calls:
            try:
                args = json.loads(c["function"]["arguments"])
                if c["function"]["name"] not in names: ok, why = False, "unknown tool " + c["function"]["name"]
                elif not isinstance(args, dict): ok, why = False, "arguments not an object"
            except (KeyError, TypeError, ValueError) as e:
                ok, why = False, "unparsable call: %s" % e
        rows.append({"prompt": p, "ok": ok, "why": None if ok else why,
                     "calls": [{"name": c.get("function", {}).get("name"), "arguments": c.get("function", {}).get("arguments")}
                               for c in calls]})
        print(("ok   " if ok else "FAIL ") + p, flush=True)
    json.dump({"passed": sum(r["ok"] for r in rows), "total": len(rows), "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
