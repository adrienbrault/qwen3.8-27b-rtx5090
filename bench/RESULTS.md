# Benchmark results

Hardware: RTX 5090 32 GB (`sm_120`), Ryzen 9 5900X, 64 GB RAM, Ubuntu 24.04, driver 595.71.05.
**GPU: +4500 MHz memory OC (16 GHz effective), 600 W, core stock.** Decode is memory-bound, so
these throughput numbers run above a stock 5090 — see [../docs/CONFIG.md](../docs/CONFIG.md#host-notes).
Model: `unsloth/Qwen3.6-27B-NVFP4` + MTP `ns=3` + vision, unless noted.
Tool: [llama-benchy](https://github.com/eugr/llama-benchy) — *not* ad-hoc curl loops, which are noisy enough to produce wrong conclusions.

> **Status (2026-07-15):** the daily is now the patched TurboQuant image with **`turboquant_4bit_nc`** KV (4-bit Keys / 4-bit Values + norm-correction) **+ `--no-async-scheduling`** — **~235K pool → 200K context** (+42% pool over `turboquant_k8v4`). This reverses the "4bit_nc destroys retrieval, 0/8" call recorded below: that 0/8 was vLLM's **async×spec-decode scheduler desync** ([vllm#42655](https://github.com/vllm-project/vllm/issues/42655)), not the 4-bit keys — `--no-async-scheduling` fixes it ([full story](../docs/HISTORY.md#status-turboquant_4bit_nc-is-the-daily-the-asyncspec-reversal)). `turboquant_k8v4` is now a decode-optimal alternative; fp8 stays the deep-context high-concurrency batch alternative.

## KV cache: turboquant_4bit_nc (daily) vs turboquant_k8v4

Both configs, same box, same session (2026-07-15), identical invocation — [llama-benchy](https://github.com/eugr/llama-benchy) 0.3.8, `--pp 512 4096 --tg 128 --concurrency 1 2 4 8 --runs 3`, util 0.94, **both with `--no-async-scheduling`**.

| | turboquant_k8v4 | **turboquant_4bit_nc** (daily) |
|---|---|---|
| KV pool | 165,274 tok | **~235,000 tok** (+42%) |
| KV memory | 4.89 GiB | ~4.89 GiB |
| KV density | 33.8K tok/GiB | **~48K tok/GiB** |
| max-model-len | 160K | **200K** (+25%) |
| decode c1 @512 (tg mean) | **137** | 133 |
| decode c2 @512 | **250** | 211 |
| decode c4 @512 | 426 | **432** |
| decode c8 @512 | **467** | 435 |
| decode c1 @4096 | **145** | 126 |
| decode c2 @4096 | 179 | 179 |
| decode c4 @4096 | 230 | 230 |
| decode c8 @4096 | **216** | 214 |
| MTP acceptance length (ns=3) | ~3.2 | ~3.2 |
| tool-eval-bench v2.1.0 | 89 | 89 |

**The honest split:** `4bit_nc` costs a **small decode tax** — −3% c1 / −7% c8 short-context (c2 is the
noisiest, −16%; c4 parity), and its worst case is **deep single-stream, −13% (c1@4096: 126 vs 145)** —
because the 4-bit-key dequant — Lloyd-Max codebook + per-GQA-head norm-correction; the inverse Hadamard is hoisted to one per-query GEMM, not per key — is more ALU
work than k8v4's cheap FP8-cast keys. From c2 up at deep context the two are within noise. In exchange
`4bit_nc` carries **+42% pool / +25% usable context**, with equal retrieval and equal MTP acceptance —
a pool-for-modest-decode trade that fits interactive coding (low concurrency, deep context).

> **Older k8v4 numbers retired.** Earlier revisions quoted `turboquant_k8v4` at decode **c1 164 @512**
> (from a standalone `k8v4-bench.json`) and a k8v4-vs-fp8 table built on it. A fresh same-session
> re-measurement did **not** reproduce the 164 — fresh k8v4 is **~137 c1 @512**. The table above uses
> the reproduced same-session figures; the 164 outlier and the derived fp8 head-to-head are dropped.
> fp8's own earlier same-session decode (for reference, not re-run under `--no-async-scheduling`):
> @512 c1 130 / c2 251 / c4 482 / c8 478 (peak c8 832); @4096 it leads from c2 up (c4@4096 461) — the
> one regime fp8 still wins is deep context at high concurrency.

### Retrieval quality (needle-in-haystack)

Plant 5-digit codes in coherent filler, exact-match. This is what *appeared* to expose
`turboquant_4bit_nc` — until we found the 0/8 was async×spec KV corruption, not the 4-bit keys.

| KV cache | 9K | 20K | 40K |
|---|---|---|---|
| `turboquant_4bit_nc` — *async scheduling ON* | **0/8 across depths (async×spec corruption, not the keys)** | | |
| **turboquant_4bit_nc** — *`--no-async-scheduling`* (daily) | **8/8** | **8/8** | **8/8** |
| fp8_e4m3 | 6/6 | 8/8 | — |
| turboquant_k8v4 | 8/8 | 8/8 | 6/6 |

`turboquant_4bit_nc` with `--no-async-scheduling` also passes **high-pressure concurrency: 90/90**
(3 rounds × 30 needles, 6 background loaders) — the exact test the "all 4-bit-KV corrupts under
concurrency" belief predicted it would fail.

**Pool ≠ usable context.** TurboQuant's continuation-prefill materializes the whole cached prefix in
bf16 (~4 KB/token transient), which **OOM-kills the engine on a single prompt far past the cap**.
The shipped config caps max-len at **200K** against the ~235K-token pool; the pool beyond the cap
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
tops the comparison. (We once thought 4-bit keys cost long-context **retrieval**, which is why we
briefly ran `turboquant_k8v4`; the "4bit_nc 0/8" was actually async×spec KV corruption — with
`--no-async-scheduling`, `4bit_nc` retrieves 8/8 and is the daily. See [KV cache](#kv-cache-turboquant_4bit_nc-daily-vs-turboquant_k8v4)
above.) One safety flag on
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
That is the number to beat — or to lose to, by however much 4-bit weights + `4bit_nc` KV cost.
A full 89-task run against that baseline is the obvious next measurement; it is not in this repo yet.

## Rejected (with numbers, so nobody redoes them)

| | result |
|---|---|
| **`turboquant_4bit_nc`** (4-bit Keys) | **UN-rejected — now the daily.** The 0/8 needle-in-haystack was async×spec KV corruption ([#42655](https://github.com/vllm-project/vllm/issues/42655)), not the keys; with `--no-async-scheduling` it scores 8/8 all depths + 90/90 concurrent, at ~235K pool. See [REJECTED.md](../docs/REJECTED.md) / [HISTORY.md](../docs/HISTORY.md#status-turboquant_4bit_nc-is-the-daily-the-asyncspec-reversal). |
| `--async-scheduling` (not passing `--no-async-scheduling`) | c4 552 → 526 on throughput **and** corrupts KV under MTP (0/8 on 4bit_nc, ~10% on k8v4). Rejected — `--no-async-scheduling` is mandatory. |
| **nvfp4-FA2** (FlashInfer FA2 nvfp4 KV) | Builds & runs byte-identical (jethac/vllm + FlashInfer #3684, JIT sm120), but loses to `4bit_nc`: stable pool 184K, decode −8..−23%, tool-eval 82, OOMs at util 0.97, 2-branch dev build + ~15min JIT. Rejected — see [REJECTED.md](../docs/REJECTED.md). |
| smaller-bit TQ presets `k3v4_nc` / `3bit_nc` | PPL delta vs bf16: k8v4 +1.17% → 4bit_nc +2.71% → k3v4_nc +10.63% → 3bit_nc +20.59%. Every preset below 4bit_nc attacks the keys. Rejected. |
| `--max-num-batched-tokens 4096` | prefill 9,607 → 2,556 (**−73%**) for +28K ctx. Rejected. |
| `VLLM_TQ_KV_SPLITS=8` | c1 143 → 132, c8 unchanged. Rejected (default 32 is right). |
| froggeric chat template | 4/4 vs bundled 4/4 on a behavioural tool-call probe. No gain. |
| DFlash | 3.3 GB draft model → 1,616-token context. Fatal on 32 GB. |
