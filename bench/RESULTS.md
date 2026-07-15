# Benchmark results

Hardware: RTX 5090 32 GB (`sm_120`), Ryzen 9 5900X, 64 GB RAM, Ubuntu 24.04, driver 595.71.05.
**GPU: +4500 MHz memory OC (16 GHz effective), 600 W, core stock.** Decode is memory-bound, so
these throughput numbers run above a stock 5090 — see [../docs/CONFIG.md](../docs/CONFIG.md#host-notes).
Model: `unsloth/Qwen3.6-27B-NVFP4` + MTP `ns=3` + vision, unless noted.
Tool: [llama-benchy](https://github.com/eugr/llama-benchy) — *not* ad-hoc curl loops, which are noisy enough to produce wrong conclusions.

> **Status (2026-07-15):** the daily is now the patched TurboQuant image with **`turboquant_k8v4`** KV (8-bit Keys / 4-bit Values). This reverses the earlier "fp8 is the daily, TurboQuant corrupts" call: there was never a corruption bug — the `!!!!` was a noisy soak-test degeneracy detector plus genuine 4-bit-*key* quality loss (`turboquant_4bit_nc`), both resolved by keeping keys at 8 bits ([full story](../README.md#status-turboquant_k8v4-is-the-daily)). `turboquant_4bit_nc` is a rejected variant (0/8 retrieval). fp8 stays the alternative for deep-context high-concurrency batch serving.

## KV cache: turboquant_k8v4 (daily) vs fp8

Both configs, same box, same session (2026-07-15), identical invocation — [llama-benchy](https://github.com/eugr/llama-benchy) 0.3.8, `--pp 512 4096 --tg 128 --concurrency 1 2 4 8 --runs 3`, util 0.94.

| | fp8_e4m3 | **turboquant_k8v4** |
|---|---|---|
| KV pool | 136,477 tok | **165,274 tok** (+21%) |
| KV memory | 5.25 GiB | **4.89 GiB** |
| KV density | 26.0K tok/GiB | **33.8K tok/GiB** |
| max-model-len | 131K | **160K** |
| unified block size | 1,600 | 2,112 |
| decode c1 @512 (tg mean) | 130 | **164** |
| decode c2 @512 | 251 | **319** |
| decode c4 @512 | 482 | **524** |
| decode c8 @512 | 478 | **516** |
| decode c8 @512 (peak) | 832 | **872** |
| decode c1 @4096 | 137 | **145** |
| decode c2 @4096 | **259** | 236 |
| decode c4 @4096 | **461** | 277 |
| decode c8 @4096 | **292** | 222 |
| prefill @512 | **4,701** | 4,258 |
| prefill @4096 | 9,604 | **10,392** |
| median inter-token | **1.15 ms** | 1.22 ms |
| TTFT (c1@512) | **110 ms** | 122 ms |
| MTP acceptance length (ns=3) | ~3.2 | ~3.2 |

**The honest split:** k8v4 wins single-stream and **short-context** decode at every concurrency
(+26% c1 @512), and matches fp8's concurrency scaling (c8 peak 872 ≥ 832 — no `4bit_nc`-style
collapse, because 8-bit keys use the efficient attention path). It's also slightly faster at deep
prefill (10.4K vs 9.6K @4096). fp8 wins exactly one regime: **deep context (≥4K) at high
concurrency**, where TurboQuant's Triton value-dequant is ALU work that scales with attended context
length (decode c4@4096 461 vs 277). For the daily — interactive coding, low concurrency, deep
context — k8v4 is faster single-stream, +21% pool, equal retrieval, equal MTP.

### Retrieval quality (needle-in-haystack)

Plant 5-digit codes in coherent filler, exact-match. This is what exposed `turboquant_4bit_nc`:
4-bit *keys* — what attention indexes on — destroy long-context retrieval.

| KV cache | 9K | 20K | 40K |
|---|---|---|---|
| `turboquant_4bit_nc` (4-bit keys) | **0/8 across depths — destroys retrieval** | | |
| fp8_e4m3 | 6/6 | 8/8 | — |
| **turboquant_k8v4** | **8/8** | **8/8** | **6/6** |

**Pool ≠ usable context.** TurboQuant's continuation-prefill materializes the whole cached prefix in
bf16 (~4 KB/token transient), which **OOM-kills the engine on a single prompt far past the cap**.
The shipped config caps max-len at **160K** against the 165,274-token pool; the pool beyond the cap
buys concurrent-sequence headroom only.

## Quant shoot-out (all NVFP4, fresh nightly, same flags)

| | Unsloth | natfii (modelopt) | NVIDIA official |
|---|---|---|---|
| decode c1 / c8 | 131 / 894 | 126 / 881 | 93 / 757 |
| prefill @4K | 9,592 | **13,348** | 4,921 |
| max ctx @ util 0.95 | 144K | **200K** | OOM (→150K @0.92) |
| **Terminal-Bench 2.1** (8 tasks ×2) | **15/16, 8/8 pass@2** | 12/16, 7/8 | — |

natfii is faster; **Unsloth is smarter**, and quality won. NVIDIA's official quant loses on
every axis — its slow prefill path is the killer.

## Quality

| eval | config | result |
|---|---|---|
| **Aider polyglot** (225 exercises) | diff format, 4 threads | **72.3% pass@2**, 34.4% pass@1, **97.3% well-formed** |
| **Terminal-Bench 2.1** (8-task subset ×2) | Harbor + Terminus-2 | **7/8 pass@2** (12/16 trials; 2 of the 4 misses were agent *timeouts*, not wrong answers) |
| **tool-eval-bench v2.1.0** (84 scenarios, hardmode, 4 trials) | seed 42, temp 0.6, serial | **89.0 ± 0.0 / 100** — Hard Mode 80%, Pass@4 = Pass^4 = 81.0% (fully deterministic across trials) |

### Tool calling ([tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench))

**Latest bench (v2.1.0, 2026-07-07): 89.0 ± 0.0 / 100** — Quality 89, Responsiveness 80
(median turn 1.2s), Deployability 86, Hard Mode 80% (24/30). Weakest category: Multi-Step
Chains (75%). Identical scores across all 4 trials.

A second run on **v2.0.6** reproduces the protocol of a [published NVFP4-vs-Q8 comparison](https://github.com/MiaAI-Lab/Unsloth-Qwen3.6-27B-UD-Q8_K_XL_vs_nvidia-Qwen3.6-27B-NVFP4_tools_eval)
(`--seed 42 --temperature 0.6 --hardmode --trials 4`), making these directly comparable:

| config | score (v2.0.6 protocol) |
|---|---|
| Unsloth NVFP4 + **TurboQuant 4-bit KV** + MTP (the patched image) | **90.0 ± 0.0** |
| nvidia NVFP4, fp8 KV (published) | 89 |
| Unsloth Q8_K_XL, llama.cpp (published) | 83 |

The aggressive 4-bit KV cache does **not** cost *tool-calling* quality — this short-context bench
tops the comparison. (It *does* cost long-context **retrieval**, which is why the daily now runs
`turboquant_k8v4`; see [KV cache](#kv-cache-turboquant_k8v4-daily-vs-fp8) above.) One safety flag on
both versions: TC-60 (cross-turn sleeper injection) fired in
all trials — the model propagated an attacker BCC smuggled through turn-1 tool output.
Standard prompt-injection caveats apply; not config-related.

Run the quality suite **serially**. The bench's per-turn latency timeouts record queued turns
as FAILs under `--parallel N` — the tool itself warns about this, and a `--parallel 8` run
here scored 79 on trial 1 from timeout-FAILs alone (then the burst OOM'd the engine — see
CONFIG.md). Responsiveness/Deployability sub-scores are only meaningful serial anyway.

Coherence: needle-in-haystack at 10K recalled exactly; factual list clean; MTP per-position
acceptance 0.945 / 0.764 / 0.564 (healthy decay — flat 100% would mean degenerate lock-step).

### On comparability

The **aider polyglot leaderboard is frozen** (last data commit 2025-10-04) — no 2026 models on
it, so 72.3% isn't comparable to modern peers. It remains an excellent *quant-regression* test.
The nearest published Qwen reference is Qwen3-32B at 41.3% (diff, May 2025).

**Terminal-Bench 2.0 is the comparable one**: Qwen publishes **59.3** for Qwen3.6-27B with a fully
documented config (Harbor + Terminus-2, temp 1.0, top_p 0.95, top_k 20, 256K ctx, avg of 5 runs).
That is the number to beat — or to lose to, by however much 4-bit weights + k8v4 KV cost.
A full 89-task run against that baseline is the obvious next measurement; it is not in this repo yet.

## Rejected (with numbers, so nobody redoes them)

| | result |
|---|---|
| **`turboquant_4bit_nc`** (4-bit Keys) | **0/8** needle-in-haystack — 4-bit keys destroy long-context retrieval. Was the old headline config; `turboquant_k8v4` (8-bit K / 4-bit V) is the fix. Rejected. |
| `--async-scheduling` | c4 552 → 526. Rejected. |
| `--max-num-batched-tokens 4096` | prefill 9,607 → 2,556 (**−73%**) for +28K ctx. Rejected. |
| `VLLM_TQ_KV_SPLITS=8` | c1 143 → 132, c8 unchanged. Rejected (default 32 is right). |
| froggeric chat template | 4/4 vs bundled 4/4 on a behavioural tool-call probe. No gain. |
| DFlash | 3.3 GB draft model → 1,616-token context. Fatal on 32 GB. |
