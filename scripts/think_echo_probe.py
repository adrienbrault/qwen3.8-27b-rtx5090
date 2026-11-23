#!/usr/bin/env python3
"""Does the served frontend render prior assistant thinking sent back by the client?

Sends one two-turn conversation four ways and compares usage.prompt_tokens: no prior thinking, prior thinking
under `reasoning_content`, under `reasoning`, and both. A field the frontend drops leaves prompt_tokens unchanged.
Also checks a tool-call turn (assistant tool_calls + reasoning_content, then a tool result).
Usage: think_echo_probe.py OUT.json [--url http://127.0.0.1:8022] [--model inkling-2-mini]
"""
import argparse
import json
import urllib.request

THOUGHT = ("The user wants the capital of France. I recall it is Paris. I should answer briefly and then wait "
           "for the follow-up question, which may ask about the population or a landmark. ") * 4


def chat(url, body):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--url", default="http://127.0.0.1:8022")
    ap.add_argument("--model", default="inkling-2-mini")
    a = ap.parse_args()

    def plain(extra):
        return [{"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris.", **extra},
                {"role": "user", "content": "And its approximate population?"}]

    tool = {"type": "function", "function": {"name": "lookup", "description": "Look up a fact.",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}

    def tooled(extra):
        return [{"role": "user", "content": "Look up the population of Paris."},
                {"role": "assistant", "content": "", **extra, "tool_calls": [{"id": "call_1", "type": "function",
                 "function": {"name": "lookup", "arguments": json.dumps({"q": "population of Paris"})}}]},
                {"role": "tool", "tool_call_id": "call_1", "name": "lookup", "content": "2.1 million (city proper)"}]

    cases = {"plain-none": plain({}), "plain-reasoning_content": plain({"reasoning_content": THOUGHT}),
             "plain-reasoning": plain({"reasoning": THOUGHT}),
             "plain-both": plain({"reasoning_content": THOUGHT, "reasoning": THOUGHT}),
             "tool-none": tooled({}), "tool-reasoning_content": tooled({"reasoning_content": THOUGHT})}
    res = {}
    for k, msgs in cases.items():
        body = {"model": a.model, "messages": msgs, "max_tokens": 1, "temperature": 0}
        if k.startswith("tool"):
            body["tools"] = [tool]
        try:
            res[k] = chat(a.url, body)["usage"]["prompt_tokens"]
        except Exception as e:  # noqa: BLE001 - record and keep going
            res[k] = f"error: {e}"
    verdict = {
        "reasoning_content_rendered": isinstance(res["plain-reasoning_content"], int)
        and res["plain-reasoning_content"] > res["plain-none"],
        "reasoning_rendered": isinstance(res["plain-reasoning"], int) and res["plain-reasoning"] > res["plain-none"],
        "tool_turn_reasoning_content_rendered": isinstance(res["tool-reasoning_content"], int)
        and res["tool-reasoning_content"] > res["tool-none"],
    }
    out = {"model": a.model, "prompt_tokens": res, **verdict}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
