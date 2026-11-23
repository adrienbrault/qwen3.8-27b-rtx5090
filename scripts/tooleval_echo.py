#!/usr/bin/env python3
"""tool-eval-bench 2.1.0 with one change: assistant turns that made tool calls carry the model's reasoning back as
`reasoning_content`, the rule tool-eval-bench 3.0.0 adopted (orchestrator._assistant_message). Everything else
(69 scenarios, scoring, sampling) is 2.1.0. Run with the 2.1.0 tool environment's python; same CLI arguments.
Prints the count of echoed assistant messages to stderr at exit so the log proves the echo happened.
"""
import atexit
import sys

import tool_eval_bench
from tool_eval_bench.runner import orchestrator

assert tool_eval_bench.__version__.startswith("2.1.0"), tool_eval_bench.__version__
_orig = orchestrator._assistant_message
_n = {"echoed": 0, "tool_turns": 0}


def _assistant_message(result):
    msg = _orig(result)
    if result.tool_calls:
        _n["tool_turns"] += 1
        if result.reasoning is not None:
            msg["reasoning_content"] = result.reasoning
            _n["echoed"] += 1
    return msg


orchestrator._assistant_message = _assistant_message
atexit.register(lambda: print(f"tooleval_echo: echoed reasoning on {_n['echoed']} of {_n['tool_turns']} "
                              "tool-call assistant turns", file=sys.stderr))

from tool_eval_bench.cli.bench import main  # noqa: E402

sys.exit(main())
