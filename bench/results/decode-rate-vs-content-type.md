# Decode rate vs content type: MTP acceptance spreads single-stream decode (2026-07-22, tier daily)

[← all results](../RESULTS.md)

Single request on the daily (:8020, tiers on), 600 completion tokens, thinking off, default sampling (temp 0.6). The only variable is what the model is asked to write:

| prompt | tok/s |
|---|---|
| "Write a short story…" (creative prose) | **82.0** |
| "Create a todo app…" (HTML/JS code) | **158.2** |

The ~2× spread comes from MTP draft acceptance alone: the `ns=4` draft head gets more tokens accepted per verify step on low-entropy, structured output. The llama-benchy matrices elsewhere in this file (~116 @pp512, ~136–140 deep-context) sample the middle of this range. A single-stream decode number on a spec-decode config is therefore a distribution over content, not a constant, so quote the workload with the number. Agent traces from the Terminal-Bench campaign show an effective 80–125 t/s including prefill share, consistent with mixed reasoning and code output.
