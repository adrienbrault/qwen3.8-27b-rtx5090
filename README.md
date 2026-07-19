# Qwen3.6-27B on a single RTX 5090 — W4A4 NVFP4, 13.5K t/s prefill, 239K KV pool, MTP K=4

Serving **Qwen3.6-27B with a 239K-token KV pool (200K usable context), ~13.5K t/s prefill, MTP speculative decoding at `ns=4`, and vision** on one 32 GB consumer GPU (RTX 5090, Blackwell `sm_120`).

The daily is the **[natfii NVFP4 W4A4](https://huggingface.co/natfii/Qwen3.6-27B-VLM-NVFP4-MTP)** checkpoint + **`fp8_e4m3` KV cache** + **FlashInfer** attention + **MTP `ns=4`**, on a patched vLLM image. The W4A4 format is what turns on Blackwell's native FP4 tensor cores — **3.4× the prefill** of the weight-only-quant daily it replaced, at equal measured quality ([why, below](#why-these-weights--and-what-actually-governs-prefill-speed)). The one patch that makes `ns=4` possible on Blackwell is [vLLM PR #42603](https://github.com/vllm-project/vllm/pull/42603) — without it, MTP + fp8 KV illegal-memory-access-crashes under any real concurrency.

## What this config optimizes for

This is a **daily driver for agentic coding** — a handful of coding agents with deep (8K–100K+) contexts, plus interactive chat and the occasional image, on one always-on box. That workload ranks the goals, and the ranking explains every choice below:

1. **Reliability over everything.** An engine that crashes mid-run or — worse — answers *fluently but wrongly* from corrupted cache is worth less than a slower one. Every config here survived a promotion gauntlet: concurrent burst battery, a fresh-deep-batch OOM trigger, needle-in-haystack recall across cache boundaries, and a 69-scenario × 2 tool-eval. Several faster configs died on that hill (a +6% pool setting, two 4-bit KV kernels, a tiered-cache patch) — the [history](docs/HISTORY.md) is mostly their graves.
2. **Trustworthy context capacity.** Agents live or die by how many deep sessions stay *warm*: a prefix-cache hit costs ~1–2 s where a cold 60K re-prefill costs ~10 s. So: the biggest KV pool that passes rule 1 (239K tokens), fp8 KV instead of denser-but-corrupting 4-bit kernels, and prefix caching on.
3. **Latency in the agent regime, not benchmark aggregate.** For agents, latency *is* mostly prefill: every fresh deep context pays it up front, and under concurrency everyone queues behind it. W4A4 tripling the prefill lane is the single biggest felt improvement in this config's history. MTP `ns=4` then roughly doubles deep single-stream decode (the "agent reading its own long context" case) even though it does nothing for shallow batch throughput.
4. **Everything on at once.** Vision, 200K context, speculative decoding, reasoning + structured outputs, tool calling — the daily runs the full stack simultaneously. No per-benchmark specialization; the numbers below are the config you'd actually run.

Non-goals: maximum batched throughput for many shallow users (a serving-farm concern — this box peaks at ~500–800 t/s aggregate anyway when streams are warm), multi-GPU, and minimum VRAM.

## Why these weights — and what actually governs prefill speed

Prefill is a compute-bound GEMM problem: thousands of tokens multiplied through the weights at once. So prefill speed is set by **which tensor-core path the quant format lets vLLM dispatch** — and that is decided by the *activation* format, not the weight bits:

| format | tensor-core path | relative GEMM rate | prefill measured here |
|---|---|---|---|
| **W4A4** (NVFP4, this daily) | native FP4 (Blackwell) | ~4× bf16 | **13.5K t/s** @8K |
| W8A8 (fp8) | fp8 | ~2× bf16 | (attention layers of the NVIDIA export) |
| **W4A16** (AutoRound, GPTQ, AWQ…) | bf16 + inline Marlin dequant | 1× bf16 − overhead | ~4.0K t/s @8K |

W4A16 keeps activations in 16-bit, so the tensor cores run plain bf16 GEMM; the 4-bit weights only save memory *bandwidth* — which is why weight-only quants decode fast but prefill no faster than bf16. W4A4 quantizes activations to FP4 on the fly and runs the whole GEMM on Blackwell's FP4 units. Decode barely differs between formats (decode is bandwidth-bound); **prefill is where the format war is won**, and for agents prefill is the latency you feel.

The quality side: W4A4's extra activation-quant error measured **≈ 1 point** of tool-eval on this checkpoint — we bounded it directly by building a [chimera checkpoint](docs/HISTORY.md) (this model's W4A4 MLPs + NVIDIA's fp8 attention) and scoring all three variants on the full 69×2 suite. natfii's calibration eats that point: it scores at parity with the best W4A16 daily we ever ran. That killed the last reason not to switch.

**The ideal model shape for this box** (32 GB Blackwell + long-context agents) — natfii is essentially it:

1. **Hybrid attention layout** (48 GDN linear-attention + 16 full-attention layers here) — linear layers pay a fixed per-sequence state instead of per-token KV, which is what makes a 200K context affordable at all on 32 GB.
2. **W4A4 NVFP4 on the MLPs at minimum** (~70% of prefill FLOPs), **with real calibration** — quantized *activations*, or the FP4 units idle.
3. **fp8-tolerant attention + fp8 KV cache** — the only KV format that survived concurrency on this hybrid; every custom 4-bit-KV kernel corrupted or crashed.
4. **MTP draft head included** (0.79 GiB → ~2× deep single-stream decode) and a compact vision tower (0.86 GiB).
5. **Clean ModelOpt/compressed-tensors export** that vLLM auto-detects — no `--quantization` flag games, no baked tokenizer quirks (see gotcha #9; this checkpoint shipped one).

## What you get

- **Qwen3.6-27B** ([natfii NVFP4 W4A4](https://huggingface.co/natfii/Qwen3.6-27B-VLM-NVFP4-MTP), auto-detected ModelOpt quant — no `--quantization` flag) over an OpenAI-compatible endpoint.
- **~13.5K t/s prefill @8K** — native Blackwell FP4 GEMM; a cold 60K-token context loads in ~10 s.
- **239K-token KV pool → 200K usable context** on `fp8_e4m3` KV — the fp8 attention path is flat with depth (no decode crater) where custom 4-bit-KV kernels crater.
- **MTP speculative decoding at `ns=4`** — draft head inside the weights — crash-free under concurrency thanks to [PR #42603](https://github.com/vllm-project/vllm/pull/42603).
- **Vision** — the model's image tower, on.
- All of it on **one 32 GB RTX 5090** (`sm_120`), memory-OC'd, 600 W.

## Benchmarks

Hardware: RTX 5090 32 GB (`sm_120`, +4500 MHz mem OC, 600 W) + Ryzen 9 5900X + 64 GB RAM, Ubuntu 24.04.
Model: Qwen3.6-27B natfii NVFP4 W4A4 + `fp8_e4m3` KV + FlashInfer 0.6.15 + MTP `ns=4` + vision, `--no-async-scheduling`, util 0.98, pool **239,436 tok**. All numbers measured on the promoted daily, 2026-07-19. (Decode/prefill rates are util-independent.)
Tool: [llama-benchy](https://github.com/eugr/llama-benchy) 0.3.8. Full detail in [bench/RESULTS.md](bench/RESULTS.md).

**Stability — the promotion gauntlet.** Zero-crash across: the ceiling battery at util 0.98 (needle-in-haystack, `pp8192×c8` + `pp30000×c8` killer shapes, 8× distinct ~34K text floods, 8× four-image vision bursts), two *simultaneous* combined waves (16 mixed requests + benchy on a cold engine), a **106-cycle overnight soak** (needle/killer/vision per cycle, zero VRAM drift), and 4 full **69×2 tool-evals** under load.

**Decode — `t/s (total)` (aggregate), `--pp 512 4096 --tg 128 --concurrency 1 2 4 8 --runs 3`:**

| decode t/s (total) | c1 | c2 | c4 | c8 | c16 |
|---|---|---|---|---|---|
| pp512 | 116 | 213 | 358 | **706** (peak 933) | 593 (peak 898) |
| pp4096 | 126 | 204 | 280 | 352 (peak 854) | — |

Over-capacity is safe: c16 (2× `max-num-seqs`) queues cleanly — the scheduler caps active streams at 8, so extra requests wait instead of destabilizing the engine. (The historical "MTP crashes at c≥16" predates PR #42603 + the seqs-8 cap.)

**Concurrency at depth — two regimes, not one number.** Batched decode at depth is *fast* on this hybrid (GDN layers pay no per-token KV reads): warm-fleet decode peaks at **732–961 t/s aggregate** (c8), ~95–136 t/s per stream. What drags *sustained* cold-context numbers down is the **prefill lane** — but W4A4 widened that lane ~3×, so the penalty shrank in proportion. The full sustained steady-state matrix (`tg 512`, aggregate t/s, peak during warm overlap in parens):

| sustained t/s (total) | c1 | c4 | c8 |
|---|---|---|---|
| pp512 | 125 | 422 (512) | **778 (961)** |
| pp4096 | 127 | 369 (495) | 605 (950) |
| pp8192 | 114 | 308 (533) | 466 (925) |
| pp30000 | 125 | 164 (481) | 149 (582) |
| pp90000 | — | 39 (263) | — |

(The previous daily managed 604/225/67 at c8 for pp512/8192/30000 on the same protocol.) Read the deep rows as prefill-lane arithmetic, not decode capability: per-stream decode peaks stay 128–136 t/s at every depth once prefills drain; pp90000×c4's 39 sustained is four cold 90K contexts sharing the ~5.3K t/s deep lane (worst TTFT ~56 s). The practical split:

- **Warm fleet** (prefix-cache hits — the normal agent-revisit case): decode aggregate ≈ the peaks, 700–930 t/s.
- **Cold fleet** (N fresh deep contexts at once): prefill-bound — ~10–13.5K t/s shared prefill lane; a cold 30K prefill now occupies ~3.0 s of it (was ~8.6 s), so the decode shadow drains 3× faster.

Measurement trap: `--tg 128` at deep contexts measures almost *only* the prefill shadow (streams never overlap in steady state) — it reports 50–56 t/s aggregate at pp30000 and looks like a regression. Use `tg ≥ 512` for steady state, and report both.

**Prefill under concurrency — a fixed shared lane, now 3.4× wider.** Prefill saturates the GPU's compute at c1, so batching adds *nothing* — aggregate stays flat and per-request throughput divides by N:

| prefill t/s | c1 | c4 aggregate (per-req) | c8 aggregate (per-req) |
|---|---|---|---|
| pp8192 | 13,315 | 13,577 (~3,340) | 13,347 (~1,670) |
| pp30000 | 10,117 | 10,001 (~2,500) | 9,878 (~1,235) |
| pp90000 | — | 5,288 (~1,320) | — |

Consequence: N simultaneous cold contexts still serialize through the lane, but the queue moves 3× faster — worst-case TTFT at c8×30K is now ~12.3 ± 6.3 s (was ~30 ± 19 s). A prefix-cache hit still skips the lane entirely.

**Long context (c1) — prefill / e2e-TTFT / decode.** Decode is **flat ~136–140 t/s from 30K → 180K** — the fp8 + FlashInfer attention path has no deep-context crater, with `ns=4` spec on top. TTFT is where W4A4 lands hardest:

| context | prefill t/s | e2e TTFT | decode t/s |
|---|---|---|---|
| 30K | 10,167 | **2.7 s** (was 7.4) | 136 |
| 90K | 5,780 | **14.1 s** (was 27.9) | 140 |
| 180K | 3,472 | **47.0 s** (was 72.4) | 138 |

(Prefill t/s falls with depth as attention's O(n²) share grows — the FP4 GEMM speedup applies to the MLP share, so the advantage narrows from ~3.4× at 8K to ~1.5× at 180K. Still faster everywhere.)

Deep context holds under concurrency too: **pp90000 × c4** runs to completion at 135 t/s per-stream decode peak — vs ~102 on the previous daily (full breakdown in the sustained matrix above).

**Quality — [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench): ~90** / 100 (full 69-scenario suite × 2 trials, ×4 independent runs, pooled mean 89.8) — statistically indistinguishable from the best W4A16 daily (87.8 pooled, same protocol; the quick-15 subset's noise band is ±7, which is why we score promotions on the full suite only). The runs double as the heaviest concurrent-load stability stress on the fix.

## Where the 31.35 GiB goes — memory budget

All numbers below are measured — the boot log (`gpu_worker` prints the breakdown at every launch, 2026-07-19 boot) plus the checkpoint's safetensors headers for the weight split:

| slice | GiB | notes |
|---|---:|---|
| **Weights, total** | **19.53** | split ↓ (19.15 on disk + loader overhead) |
| · language model | 12.76 | 64 hybrid layers (48 GDN + 16 full-attention), NVFP4 W4A4 + per-block scales |
| · embeddings | 2.37 | bf16 — not quantized |
| · lm_head | 2.37 | bf16 — embed+head = 4.7 GiB, a **24% big-vocab tax** on the weight budget |
| · vision tower | 0.86 | bf16, always loaded (even text-only requests) |
| · MTP drafter | 0.79 | shipped less-quantized than the previous daily's (0.28) — the price of the ns=4 head that ~2×'s deep decode |
| **KV pool** | **8.93** | **= 239,436 tokens** @ ~39 KiB/tok (fp8 attention KV + GDN state pages, one unified pool) |
| **Peak-activation reserve** | 1.89 | sized by profiling at `mnbt` 4096 |
| **CUDA graphs + non-torch** | 0.40 | |
| **util 0.98 budget** | **30.72** | of 31.35 usable; ~130–190 MiB steady-state free after autotune workspaces allocate — validated at exactly this margin, see gotcha #8 |

Honest trade vs the previous daily: natfii's weights are **1.8 GiB heavier** in VRAM (fatter MTP head + FP4 scale tensors), so even at util 0.98 the pool is **239K vs the old 270K** (−11%). We took it: prefill 3.4×, deep-concurrent throughput 2.2×, equal quality. Capacity you re-prefill 3× faster is worth more than capacity you wait on.

What each token costs and what a cache hit is worth (60K-token context, measured):

| tier | capacity | revisit cost | status |
|---|---|---|---|
| GPU pool (vLLM prefix cache) | 239,436 tok | **~1–2 s** (≈ decode time only) | production |
| host RAM (LMCache L1, 24 GiB pinned) | ~245K tok @ ~98 KiB/tok serialized | ~2 s | **works with 3 local patches — pre-production, see below** |
| NVMe (LMCache L2) | ~640K tok per 60 GiB | **~5–7.5 s** (survives restarts) | **works with 3 local patches — pre-production, see below** |
| miss → full re-prefill | — | **~10 s** (was ~23 s on the W4A16 daily — the FP4-GEMM dividend) | the thing caches exist to avoid |

Notes that save you from wrong conclusions:

- **util is the only pool lever** (+~8.4K tok per 0.01: 222,535 @0.96 → 239,436 @0.98 measured); `max-num-seqs` provably isn't on this hybrid (`align` packs GDN state *into* the unified pool, consumed per **active** request — seqs 4 vs 8 gave the identical pool).
- The in-pool ~39 KiB/tok is an *effective average*: full-attention layers pay per-token fp8 KV; GDN layers pay a fixed per-sequence state that amortizes with depth. Serialized tiers pay ~98 KiB/tok because LMCache ships whole 1616-token unified blocks (attention KV + state page + metadata, padded to full block).
- Effective deep concurrency is **pool-bound, not `max-num-seqs`-bound**: 4 × ~70K-token agent sessions fill the pool; request 5 queues.
- The RAM/NVMe tiers now **pass the cross-restart needle test and score quality parity (88 vs 86–90 controls on the full 69×2)** — but only with **three local patches**, none upstream yet: two on LMCache `main` (a stride-aware regroup of the fp8 backend's 16-token kernel pages into logical hybrid pages — releases through 0.5.1 silently store *nothing* on this layout, and our earlier [withdrawn kernel patch](patches/README.md#lmcache-format-10-kernel-patch-separate-project) corrupted state) and one on **vLLM itself**: with any KV connector configured *and* MTP enabled on a hybrid model, the scheduler's connector-path prefix-hit lookup mixes an EAGLE-adjusted attention hit with an unadjusted Mamba hit and takes `max()` — leaving one **allocated-but-never-filled attention block** at every local cache hit (cost: ~10 eval points, in either connector role, even at concurrency 1). One narrow MTP-rehit edge case is still under investigation, so the tiered profile is **pre-production**. Full story in [docs/LMCACHE.md](docs/LMCACHE.md). Whatever you run: **needle-test across a restart before trusting any external KV tier on a hybrid model** — hit counters and coherent output do not prove fidelity.

## Why it needs a patch: MTP × fp8-KV × Blackwell crashes on stock vLLM

> ⚠️ **`ns≥2` MTP with `fp8_e4m3` KV on `sm_120` is a 100%-reproducible illegal-memory-access under concurrency** — a known, still-open upstream bug ([vllm#40756](https://github.com/vllm-project/vllm/issues/40756), same Qwen3.6-27B model; [vllm#35288](https://github.com/vllm-project/vllm/issues/35288) "MTP corrupted output at concurrency ≥ 4"). It crashes at `rejection_sampler.py:267 parse_output → cudaErrorIllegalAddress`. Single-stream and `ns=1` are both clean; `CUDA_LAUNCH_BLOCKING=1` masks it (→ a timing race).

The root cause ([PR #42603](https://github.com/vllm-project/vllm/pull/42603)): the MTP draft loop in `vllm/v1/spec_decode/llm_base_proposer.py` writes shared cudagraph input buffers (`input_ids`, `hidden_states`), then immediately launches the draft-model forward that reads them — **without a stream sync**. CUDA is async, so under concurrency the draft FlashInfer kernels observe stale / partially-written buffers → illegal access. The fix is one line:

```python
self.input_ids[:batch_size] = input_ids
self.hidden_states[:batch_size] = hidden_states
torch.accelerator.current_stream().synchronize()   # PR #42603
```

That's it. It restores `ns=4` at full pool, is **perf-neutral**, and is validated crash-free across the concurrent + deep-context + tool-eval load that reliably crashed before (see [Benchmarks](#benchmarks)). [`patches/install_pr42603_sync.py`](patches/install_pr42603_sync.py) is an idempotent, `ast`-verified graft of it onto the installed vLLM tree.

Plus one quality graft: **[PR #44993](https://github.com/vllm-project/vllm/pull/44993)** — structured output (`response_format` json_schema) silently returned empty `content` under a reasoning model (the schema JSON leaked into `reasoning_content`). Fixed with the `--structured-outputs-config` flag below.

## Quick start

```bash
# build the patched image (~1 min of pure-Python patches on top of the vLLM base;
# the FlashInfer 0.6.15 step JIT-compiles its kernels on first run, so mount its cache)
cd patches && docker build -t vllm-qwen36:patched .

# serve
./scripts/serve.sh
```

Then `http://localhost:8020/v1` speaks OpenAI. Every flag is explained inline in [`scripts/serve.sh`](scripts/serve.sh) and in [Config essentials](#config-essentials) below.

## What's in the patch stack

The daily uses these three; the image also carries an (unused-by-this-config) TurboQuant 4-bit-KV stack documented separately in [docs/HISTORY.md](docs/HISTORY.md).

| patch | what it does |
|---|---|
| [`install_pr42603_sync.py`](patches/install_pr42603_sync.py) — [PR #42603](https://github.com/vllm-project/vllm/pull/42603) | **The fix that makes MTP `ns=4` usable on Blackwell.** One `torch.accelerator.current_stream().synchronize()` in the MTP draft loop, after the cudagraph-buffer writes and before the draft forward — closes the stale-buffer race that IMA-crashes `ns≥2` + fp8 KV under concurrency ([#40756](https://github.com/vllm-project/vllm/issues/40756), [#35288](https://github.com/vllm-project/vllm/issues/35288)). Perf-neutral. [Validated numbers](bench/RESULTS.md#mtp-k4-restored-on-blackwell--the-fp8-kv-spec-decode-crash-pr-42603). |
| FlashInfer 0.6.15 (Dockerfile pip step) | Latest FlashInfer, carrying the `sm_120` GDN/TMA fixes. cu130 AOT cubin/jit-cache isn't published for .15, so the image drops the mismatched 0.6.13 caches and lets 0.6.15 JIT-compile at runtime — **mount `/root/.cache/flashinfer`** (one build, warm forever). |
| [PR #44993 graft](https://github.com/vllm-project/vllm/pull/44993) — `v1/structured_output/__init__.py`, `v1/core/sched/scheduler.py` | **Structured output that survives thinking.** With a reasoning model, `response_format` json_schema + thinking-on returned EMPTY `content` (the schema JSON leaked into `reasoning_content`) — `should_advance`'s delta window skips `</think>` when MTP rejects drafts, so the grammar never re-engages. Needs `--structured-outputs-config '{"backend":"xgrammar","reasoning_parser":"qwen3","enable_in_reasoning":false}'`. Two pure-Python files. |

## Config essentials

`./scripts/serve.sh` runs the daily (flags also annotated inline there). The load-bearing ones:

- **No `--quantization` flag** — the natfii checkpoint is a ModelOpt NVFP4 export; vLLM auto-detects it and dispatches the CUTLASS FP4 GEMM kernels (`FlashInferCutlassNvFp4LinearKernel`). Forcing a flag here selects the wrong kernel path.
- `-e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728` — caps FlashInfer's lazily-allocated autotune workspace at 128 MiB; part of what makes util 0.98 survivable (gotcha #8). Perf-neutral in every regime we measured.
- `--kv-cache-dtype fp8_e4m3` + `-e VLLM_ATTENTION_BACKEND=FLASHINFER` — the flat-with-depth attention path; `e5m2` is **not** usable on this checkpoint (vLLM: "fp8_e5m2 not supported with fp8 checkpoints").
- `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":4}'` — MTP `ns=4`. **Requires the [PR #42603](https://github.com/vllm-project/vllm/pull/42603) graft** or it IMA-crashes under concurrency.
- **`--no-async-scheduling` — keep it.** MTP emits a multi-token verify batch every step; vLLM's async scheduler desyncs its request-ID→batch-row mapping under spec decode ([vllm#42655](https://github.com/vllm-project/vllm/issues/42655)) and corrupts KV. Async-off is the documented fix.
- `--mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill` — hybrid-model prefix caching; **`align` is load-bearing for the 239K pool** (it packs the GDN/Mamba state into the unified KV pool). Mode `all` is same speed / same pool and does **not** avoid the crash — don't bother.
- `--gpu-memory-utilization 0.98 --max-model-len 200000 --max-num-batched-tokens 4096` — let vLLM profile the pool; don't hand-set `--kv-cache-memory` (its "fully utilize" hint ignores warmup transients and OOMs). util is the only pool lever here (~+8.4K tok/0.01 → 239,436 at 0.98); `mnbt` doesn't change the pool, and `mnbt 4096` avoids a ~9% deep-prefill slowdown that `mnbt 8192` incurs. **The util ceiling is model-specific, not a constant** — 0.98 killed the previous (heavier-shaped) daily at serve time but passes the full adversarial battery on this one *with* the 128 MiB workspace cap and a boot-time shape pre-warm; re-validate against gotcha #8's killer shape if you change anything. 0.96 (222K) is the validated fallback.
- `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml` — `qwen3_xml` is correct; `hermes` silently drops tool calls.
- `--override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}'` + `--default-chat-template-kwargs '{"preserve_thinking":true}'` — keep historical `<think>` blocks across turns. **Caveat:** the *client* must resend prior reasoning in the **`reasoning`** field (not the deprecated `reasoning_content`, which vLLM ignores on input).
- `--structured-outputs-config '{"backend":"xgrammar","reasoning_parser":"qwen3","enable_in_reasoning":false}'` — enables the reasoning gate the [#44993](https://github.com/vllm-project/vllm/pull/44993) graft needs. Give it an adequate `max_tokens` budget (reasoning + JSON) or it truncates mid-think and looks empty.
- `--limit-mm-per-prompt '{"image":4,"video":0}'` — vision on.

**Verify container identity after launch.** Confirm the startup log reports a **~239K-token KV pool** (at util 0.98). A wrong pool size means a preset/dtype/align mismatch reading as a perfectly healthy server at the *wrong* config.

## Gotchas that bite during setup

1. **MTP `ns≥2` needs [PR #42603](https://github.com/vllm-project/vllm/pull/42603) or it IMA-crashes under concurrency.** Single-stream and `ns=1` pass every test and hide it; `CUDA_LAUNCH_BLOCKING=1` masks it. **Load-test with 3+ parallel streams on day one** — that's the only thing that reproduces it.
2. **`--no-async-scheduling` is mandatory with MTP.** Async scheduling desyncs the request-ID→batch-row mapping under spec decode ([#42655](https://github.com/vllm-project/vllm/issues/42655)) and corrupts KV.
3. **Verify the ~239K KV pool in the launch log.** A silent config fallback (wrong preset, `align` off, wrong image) looks like a healthy server at the wrong pool.
4. **flashinfer JIT eats all host RAM (non-nightly images).** Any non-nightly vLLM image on `sm_120` JIT-compiles CUTLASS kernels on the first forward with unbounded `nvcc` parallelism — multi-GB per job, reads as a mystery "hang" or whole-host livelock. Cap it (`MAX_JOBS=4` + `FLASHINFER_NUM_COMPILE_JOBS=4`) and **mount a persistent `/root/.cache/flashinfer`** (one build, warm forever).
5. **vLLM's prefix-cache metric lies on this model.** `vllm:prefix_cache_hits_total` / "Prefix cache hit rate: 0.0%" report **0% while the cache works**. Don't debug the counter — time a repeated prompt ([`bench/prefix_probe.py`](bench/prefix_probe.py)).
6. **Validate coherence via raw `/v1/completions`.** The chat endpoint's reasoning parser swallows degenerate output as *empty content*, so a broken model looks "fine but quiet." Tells: constant-token output, or **flat 100% MTP acceptance** (draft and verify locked in step).
7. **Re-verify after every vLLM bump.** These grafts are version-sensitive; a nightly that moves `llm_base_proposer.py` will fail the `ast` check at build (loud) or, worse, shift the anchor.
8. **The util ceiling is set by *lazy autotune workspace*, boot-margin probes cannot see it, and it is model-specific.** The first time the engine meets a genuinely new batch shape (e.g. a fresh 8×8K concurrent prefill+decode wave), the fp4-GEMM/FlashInfer autotuner allocates its benchmark workspace **at serve time**: measured ~266 MiB for `mnbt` 4096 shapes, ~486 MiB for 8192 shapes, on top of allocator fragmentation. This OOM-killed the previous (W4A16) daily at util 0.98 mid-traffic, 100% reproducibly (`benchy --pp 8192 --concurrency 8` is the reliable trigger) — even though near-pool-full text bursts and 8×4-image vision bursts all passed. The current daily *does* run 0.98, and earned it: `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728` caps the workspace, `serve.sh` pre-warms the killer shape at boot so the allocation happens before real traffic, and the config passed a 0.98 battery ending in two *simultaneous* waves of 8 deep-text floods + 8 four-image vision requests + `pp8192×c8` benchy on a cold engine (steady-state floor: ~130 MiB free, stable across waves). Traps when you probe a ceiling yourself: (a) identical prompts are silently collapsed by prefix caching — use distinct prompts; (b) text-only bursts miss the vision-encoder transient; (c) **prefill-style bursts miss the autotune transient — include a fresh deep-prefill × full-decode shape (`pp8192 × c8`), and fire your stressors *concurrently*, not sequentially.** `mnbt` 8192 needs more margin (≲0.94) — its bigger shapes want the bigger workspace.
9. **Check `tokenizer.json` for baked truncation.** This checkpoint shipped with `"truncation": {"max_length": 8192}` left over from calibration — a silent poison pill: text works, single small images work, but any multimodal request expanding past 8192 tokens hard-fails with a processor mismatch (HTTP 400). `serve.sh` verifies and nulls it at every launch, because **re-downloading the model reintroduces the bug**.

## Daily lineage — what each daily was, and why the next took over

Newest first. Every switch is documented with numbers in [docs/HISTORY.md](docs/HISTORY.md) and [bench/RESULTS.md](bench/RESULTS.md).

| daily | weights · KV | pool | why it took over |
|---|---|---|---|
| **natfii NVFP4 W4A4 · fp8_e4m3** + FlashInfer + MTP `ns=4` **(current, 2026-07-19)** | NVFP4 W4A4 · fp8 | **~239K** @0.98 | **Prefill 3.4×** (13.5K vs 4.0K t/s @8K — native Blackwell FP4 GEMM vs Marlin dequant), deep-concurrent sustained **2.2×** (148 vs 67 t/s at pp30K×c8 tg512), cold 60K context 10 s vs 23 s — at **equal 69×2 quality** (~90, 4 trials each side; the W4A4 activation cost was bounded at ≈1 pt via a chimera A/B and natfii's calibration covers it). Survived the full promotion gauntlet incl. a 106-cycle soak and a 0.98 combined-wave battery. Pool is 11% smaller than AR's 270K (heavier MTP head + FP4 scales) — traded for re-prefilling 3× faster. |
| Lorbus INT4-AutoRound · fp8_e4m3 + FlashInfer + MTP `ns=4` (2026-07-18) | INT4-AutoRound · fp8 | ~270K @0.96 | Flat deep decode (fp8+FlashInfer has **no** decode crater at depth, where the custom TurboQuant kernel drops); biggest pool ever; **MTP `ns=4`** restored by [PR #42603](https://github.com/vllm-project/vllm/pull/42603); tool-eval 90; dropped the experimental TurboQuant KV kernel for the **battle-tested fp8** path. (A one-day 0.98/287K promotion was reverted the same night: serve-time autotune OOM — gotcha #8.) |
| turboquant_4bit_nc (NVFP4) + MTP `ns=3` (2026-07-15) | NVFP4 · TQ 4-bit K/V | ~235K | +42% pool over k8v4, once the "4bit_nc destroys retrieval" **0/8** was traced to the async×spec KV confound and fixed with `--no-async-scheduling`. Decode still craters at deep single-stream (the custom-kernel cost). |
| turboquant_k8v4 (NVFP4) | NVFP4 · TQ 8-bit K/4-bit V | ~165K | +21% pool over fp8 at fp8-equal retrieval quality (8-bit keys). |
| fp8_e4m3 (stock nightly) | NVFP4 · fp8 | ~136K | The original battle-tested baseline — flat deep decode, no patches, smallest pool. |

The arc, compressed: the fp8 baseline's **flat-decode virtue** survived every generation; PR #42603 added working `ns=4`; AutoRound added quality and pool; and NVFP4 W4A4 finally cashed in the GPU's native FP4 compute — the first daily where *prefill* got a generational jump instead of decode or capacity.

## How we got here / what didn't work

- **[docs/HISTORY.md](docs/HISTORY.md)** — the full path to this config, including the multi-round bisection that localized the `ns=4` crash: barriers placed *around* the proposer in `gpu_model_runner` all fired and still crashed (the race is *inside* the proposer loop); `--mamba-cache-mode all` still crashed (so it isn't the fused Mamba postprocess); a draft-token sanitizer never fired (the fault was never a bad token) — until the upstream trail led to [PR #42603](https://github.com/vllm-project/vllm/pull/42603). Also documents the earlier TurboQuant 4-bit-KV and NVFP4 work that this repo previously shipped as the daily.
- **[docs/REJECTED.md](docs/REJECTED.md)** — everything tried and rejected, with the number that killed it. Read it before "improving" the config.

## License

MIT (see [LICENSE](LICENSE)) for the original work here — docs, benchmarks, scripts, and the grafts. The vLLM files the patches modify stay under **Apache-2.0**; see [THIRD_PARTY.md](THIRD_PARTY.md).

## Credits

- [vLLM](https://github.com/vllm-project/vllm) and [PR #42603](https://github.com/vllm-project/vllm/pull/42603) — the draft-loop sync that makes MTP `ns=4` usable on Blackwell.
- [natfii](https://huggingface.co/natfii/Qwen3.6-27B-VLM-NVFP4-MTP) for the Qwen3.6-27B W4A4 NVFP4 export (ModelOpt 0.43) — the current daily's weights.
- [Lorbus](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) for the Qwen3.6-27B INT4-AutoRound quant — the previous daily, still the W4A16 reference.
- [llama-benchy](https://github.com/eugr/llama-benchy), [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) for the measurements.
