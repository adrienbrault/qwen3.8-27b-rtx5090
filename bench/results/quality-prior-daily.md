# Quality, prior daily (NVFP4 + TurboQuant)

[← all results](../RESULTS.md)

| eval | config | result |
|---|---|---|
| **Aider polyglot** (225 exercises) | diff format, 4 threads | **72.3% pass@2**, 34.4% pass@1, **97.3% well-formed** |
| **Terminal-Bench 2.1** (8-task subset ×2) | Harbor + Terminus-2 | **7/8 pass@2** (12/16 trials; 2 of the 4 misses were agent *timeouts*, not wrong answers) |
| **tool-eval-bench v2.1.0** (84 scenarios, hardmode, 4 trials) | seed 42, temp 0.6, serial | **89.0 ± 0.0 / 100** — Hard Mode 80%, Pass@4 = Pass^4 = 81.0% (fully deterministic across trials) |

## Tool calling ([tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench))

**That era's bench (v2.1.0, 2026-07-07): 89.0 ± 0.0 / 100**, with Quality 89, Responsiveness 80 (median turn 1.2s), Deployability 86, and Hard Mode 80% (24/30). The weakest category was Multi-Step Chains (75%). Scores were identical across all 4 trials, under a serial, seeded protocol, unlike the current daily's sampled parallel-8 runs ([cross-trial stats at the top of this file](tool-eval-cross-trial-stats.md)).

A second run on v2.0.6 reproduces the protocol of a [published NVFP4-vs-Q8 comparison](https://github.com/MiaAI-Lab/Unsloth-Qwen3.6-27B-UD-Q8_K_XL_vs_nvidia-Qwen3.6-27B-NVFP4_tools_eval) (`--seed 42 --temperature 0.6 --hardmode --trials 4`), making these directly comparable:

| config | score (v2.0.6 protocol) |
|---|---|
| Unsloth NVFP4 + **TurboQuant 4-bit KV** + MTP (the patched image) | **90.0 ± 0.0** |
| nvidia NVFP4, fp8 KV (published) | 89 |
| Unsloth Q8_K_XL, llama.cpp (published) | 83 |

The aggressive 4-bit KV cache does not cost tool-calling quality, and this short-context bench tops the comparison. 4-bit keys were once thought to cost long-context retrieval, which is why `turboquant_k8v4` ran briefly; the "4bit_nc 0/8" was actually async×spec KV corruption, and with `--no-async-scheduling`, `4bit_nc` retrieves 8/8 and became the then-daily. See [KV cache](kv-cache-turboquant-4bit-nc-vs-k8v4.md) above. One safety flag applies to both versions: TC-60 (cross-turn sleeper injection) fired in all trials, with the model propagating an attacker BCC smuggled through turn-1 tool output. Standard prompt-injection caveats apply, and this is not config-related.

Run the quality suite **serially**. The bench's per-turn latency timeouts record queued turns as FAILs under `--parallel N`; the tool itself warns about this, and a `--parallel 8` run here scored 79 on trial 1 from timeout-FAILs alone, after which the burst OOM'd the engine (see CONFIG.md). Responsiveness and Deployability sub-scores are only meaningful when run serially.

Coherence: needle-in-haystack at 10K recalled exactly, the factual list was clean, and MTP per-position acceptance measured 0.945 / 0.764 / 0.564, a normal decay (a flat 100% would indicate degenerate lock-step).

## On comparability

The **aider polyglot leaderboard is frozen** (last data commit 2025-10-04), with no 2026 models on it, so 72.3% is not comparable to modern peers. It remains a useful quant-regression test. The nearest published Qwen reference is Qwen3-32B at 41.3% (diff, May 2025).

**Terminal-Bench 2.0** is the comparable benchmark: Qwen publishes 59.3 for Qwen3.6-27B with a fully documented config (Harbor + Terminus-2, temp 1.0, top_p 0.95, top_k 20, 256K ctx, avg of 5 runs). That is the reference number, and the gap to it measures what 4-bit weights plus `4bit_nc` KV cost. A full 89-task run against that baseline is the obvious next measurement, and it is not in this repo yet.
