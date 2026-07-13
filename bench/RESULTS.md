# Benchmark results

Hardware: RTX 5090 32 GB (`sm_120`), Ryzen 9 5900X, 64 GB RAM, Ubuntu 24.04, driver 595.71.05.
**GPU: +4500 MHz memory OC (16 GHz effective), 600 W, core stock.** Decode is memory-bound, so
these throughput numbers run above a stock 5090 — see [../docs/CONFIG.md](../docs/CONFIG.md#host-notes).
Model: `unsloth/Qwen3.6-27B-NVFP4` + MTP `ns=3` + vision, unless noted.
Tool: [llama-benchy](https://github.com/eugr/llama-benchy) — *not* ad-hoc curl loops, which are noisy enough to produce wrong conclusions.

> **Status (2026-07-13):** we now run **stock nightly + fp8 KV** as the daily and treat the TurboQuant image as experimental — 4-bit KV intermittently corrupts under real agent tool-calling (constant `!!!!`, 0% draft acceptance; [details](../README.md#status--2026-07-13-fp8-is-the-daily-the-turboquant-image-is-experimental)). These throughput and quality numbers still stand — the corruption is intermittent and never surfaced on any benchmark here — but read the fp8/TurboQuant comparison below as "faster vs reliable," not "old vs new default."

## KV cache: TurboQuant 4-bit vs fp8

Identical image, flags, and tool; same day.

| | fp8_e4m3 | **turboquant_4bit_nc** |
|---|---|---|
| KV pool | 172,000 | **261,333** |
| KV density | 26.8K tok/GiB | **47.8K tok/GiB** |
| decode c1 | 129 | **143** |
| decode c2 | 253 | 251 |
| decode c4 | 492 | **552** |
| decode c8 | **868** | 540 |
| prefill @512 | 4,466 | 4,466 |
| prefill @4096 | 9,607 | **10,222** |
| MTP acceptance | ~60% | **75.8%** |

TurboQuant's Triton decode kernel saturates around 4 concurrent and plateaus; its 4-bit dequant
is ALU work that flash-attn's fp8 path gets free in hardware. On throughput alone: single-user →
TurboQuant, 8+ concurrent → fp8 (+61%). In practice we run **fp8 everywhere** — TurboQuant's
intermittent corruption under real agent sessions outweighs its ~10% single-stream edge (see status
note above).

**Pool ≠ usable context.** These pools were measured at util 0.95 / 240K max-len, but TurboQuant's
continuation-prefill materializes the whole cached prefix in bf16 (~4 KB/token transient), which
**OOM-kills the engine on any single prompt past ~160K**. The shipped config caps max-len at
**150K** (147K verified end-to-end); the pool beyond that buys concurrent-sequence headroom only.

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
| Unsloth NVFP4 + **4-bit TurboQuant KV** + MTP (the experimental image) | **90.0 ± 0.0** |
| nvidia NVFP4, fp8 KV (published) | 89 |
| Unsloth Q8_K_XL, llama.cpp (published) | 83 |

The aggressive 4-bit KV cache does **not** cost tool-calling quality — this config tops the
comparison. One safety flag on both versions: TC-60 (cross-turn sleeper injection) fired in
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
That is the number to beat — or to lose to, by however much 4-bit weights + 4-bit KV cost.
A full 89-task run against that baseline is the obvious next measurement; it is not in this repo yet.

## Rejected (with numbers, so nobody redoes them)

| | result |
|---|---|
| `--async-scheduling` | c4 552 → 526. Rejected. |
| `--max-num-batched-tokens 4096` | prefill 9,607 → 2,556 (**−73%**) for +28K ctx. Rejected. |
| `VLLM_TQ_KV_SPLITS=8` | c1 143 → 132, c8 unchanged. Rejected (default 32 is right). |
| froggeric chat template | 4/4 vs bundled 4/4 on a behavioural tool-call probe. No gain. |
| DFlash | 3.3 GB draft model → 1,616-token context. Fatal on 32 GB. |
