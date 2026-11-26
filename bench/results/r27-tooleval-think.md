# r27 tool-eval with earlier reasoning sent back (2026-11-23, results `2026-11-23-r27-tooleval-think`)

[← all results](../RESULTS.md)

**Superseded 2026-11-26.** This boot ran with BA4 inactive (0 of 96 rank-layers, after the 2026-11-20 card slot swap), so it was not the served configuration. The served-configuration row is [r27 tool-eval with BA4](r27-tooleval-ba4.md): 95.2 ± 1.5.

2026-11-23 22:49 to 23:01 UTC, raw records in [`2026-11-23-r27-tooleval-think/`](2026-11-23-r27-tooleval-think/), driver [`scripts/r27-tooleval-think.sh`](../../scripts/r27-tooleval-think.sh). One boot of the served launcher ([`scripts/serve-v031-served-daily.sh`](../../scripts/serve-v031-served-daily.sh), image `vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan`) with a freshly compiled `torch.compile` artifact, nothing else using the server.

Every tool-eval row in this repository before this date ran tool-eval-bench 2.1.0, which resends conversation history without each assistant turn's reasoning. The Qwen3.8 template keeps reasoning on every earlier assistant turn and renders an empty `<think>` block when a client sends none. tool-eval-bench 3.0.0 sends tool-calling turns' reasoning back as `reasoning_content`. Arm B is 2.1.0 with only that change ([`scripts/tooleval_echo.py`](../../scripts/tooleval_echo.py)), so A against B isolates the reasoning echo on the same 69 scenarios.

[tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench), 69 scenarios × 4 trials, temperature 0.6, top-p 0.95, top-k 20, 8 in parallel.

| arm | score (mean ± stddev of 4 trials), median | trial points of 138, mean |
| --- | ---: | ---: |
| A: tool-eval-bench 2.1.0 | 88.5 ± 1.0, 88.0 | 122.5 |
| B: 2.1.0 + reasoning echo (373 of 373 tool-calling turns echoed) | 90.0 ± 1.8, 90.0 | 124.2 |
| C: tool-eval-bench 3.0.0 | 94.5 ± 1.7, 94.0 | 130.8 |

- The reasoning echo moved the 2.1.0 score by +1.5, within one trial standard deviation of either arm. Most of the difference between 2.1.0 and 3.0.0 comes from 3.0.0's other changes.
- `echo-probe.json` ([`scripts/think_echo_probe.py`](../../scripts/think_echo_probe.py)): one conversation's prompt tokens without and with earlier reasoning. `reasoning_content` and `reasoning` each add the reasoning to the prompt, on a plain earlier turn (38 → 187 tokens) and on a tool-calling turn (324 → 473).
- A is the same-boot control for the 2026-10-20 served row (89.6 ± 1.8 over 8 trials, [r27 served quality](r27-served-quality.md)).
