#!/usr/bin/env python3
"""Small live arithmetic/tool smoke, with raw responses retained for inspection."""
import argparse
import json
import math
import pathlib
import urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def ask(name, path, **body):
        data = {"model": a.model, "temperature": 0, **body}
        req = urllib.request.Request(a.url + path, json.dumps(data).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as response:
            result = json.load(response)
        (out / (name + ".json")).write_text(json.dumps(result, indent=2))
        return result["choices"][0]

    choice = ask("arithmetic", "/v1/chat/completions", max_tokens=128,
                 chat_template_kwargs={"enable_thinking": False},
                 messages=[{"role": "user", "content": "What is 19 + 23? Reply with only the integer."}])
    assert (choice["message"].get("content") or "").strip() == "42", choice
    tool = {"type": "function", "function": {"name": "record_reading",
            "description": "Record the supplied temperature reading.",
            "parameters": {"type": "object", "properties": {
                "value": {"type": "integer"}, "unit": {"type": "string"}},
                "required": ["value", "unit"]}}}
    choice = ask("tool", "/v1/chat/completions", max_tokens=256,
                 chat_template_kwargs={"enable_thinking": False}, tools=[tool],
                 messages=[{"role": "user", "content": "Call record_reading with value 17 and unit C. Use the tool; do not answer in prose."}])
    calls = choice["message"].get("tool_calls") or []
    assert len(calls) == 1, choice
    function = calls[0]["function"]
    assert function["name"] == "record_reading", choice
    assert json.loads(function["arguments"]) == {"value": 17, "unit": "C"}, choice
    choice = ask("completion", "/v1/completions", max_tokens=128, logprobs=2,
                 prompt="Explain why binary search takes logarithmic time.\n")
    scores = choice["logprobs"]["token_logprobs"]
    assert len(scores) >= 16 and all(x is not None and math.isfinite(x) for x in scores), choice
    print("SMOKE PASS: arithmetic, parsed tool arguments, finite decode logprobs", flush=True)


if __name__ == "__main__":
    main()
