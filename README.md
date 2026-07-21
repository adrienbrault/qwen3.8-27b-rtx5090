# Qwen3.6-27B on a single RTX 5090 — W4A4 NVFP4, 13.5K t/s prefill, ~2.6M tokens of tiered KV, MTP K=4

Serving **Qwen3.6-27B at 200K context with ~13.5K t/s prefill, MTP speculative decoding at `ns=4`, vision, and a three-tier KV cache — 214K tokens on-GPU, ~245K in pinned DRAM, ~2.13M on NVMe** — on one 32 GB consumer GPU (RTX 5090, Blackwell `sm_120`). The NVMe tier survives restarts, so yesterday's agent session is still warm this morning.

The daily is the **[natfii NVFP4 W4A4](https://huggingface.co/natfii/Qwen3.6-27B-VLM-NVFP4-MTP)** checkpoint + **`fp8_e4m3` KV cache** + **FlashInfer** attention + **MTP `ns=4`** + **[LMCache](https://github.com/LMCache/LMCache) DRAM/NVMe offload**, on a patched vLLM image. Two things carry it:

- **W4A4 turns on Blackwell's native FP4 tensor cores** — **3.4× the prefill** of the weight-only-quant daily it replaced, at equal measured quality ([why, below](#why-these-weights--and-what-actually-governs-prefill-speed)). The patch that makes `ns=4` possible at all on Blackwell is [vLLM PR #42603](https://github.com/vllm-project/vllm/pull/42603); without it, MTP + fp8 KV illegal-memory-access-crashes under any real concurrency.
- **The KV tiers turn eviction into a 2–7 s reload instead of an 11–13 s re-prefill**, and turn a restart from a total cache loss into a warm start. On a 16-task SWE-Bench-Verified run at 4 concurrent agents that's **3.4× the wall-clock throughput** ([A/B](#what-removing-lmcache-changes)) — agents resend their whole transcript every step, so almost every request is a long prefix revisit. Getting them *faithful* on an fp8 hybrid took **[six local patches](patches/lmcache/README.md)** — four on LMCache, two on vLLM. Unpatched, this profile is worse than no cache at all: it stores wrong-addressed pages and restores garbage recurrent state, with fluent output and zero errors logged.

If you want the engine without the tiers — bigger hot pool, no sidecar, no local LMCache patches — that's [`scripts/serve-plain.sh`](scripts/serve-plain.sh), and the exact trade is spelled out in [What removing LMCache changes](#what-removing-lmcache-changes).

## What this config optimizes for

This is a **daily driver for agentic coding** — a handful of coding agents with deep (8K–100K+) contexts, plus interactive chat and the occasional image, on one always-on box. That workload ranks the goals, and the ranking explains every choice below:

1. **Reliability over everything.** An engine that crashes mid-run or — worse — answers *fluently but wrongly* from corrupted cache is worth less than a slower one. Every config here survived a promotion gauntlet: concurrent burst battery, a fresh-deep-batch OOM trigger, needle-in-haystack recall across cache boundaries, and a 69-scenario × 2 tool-eval. Several faster configs died on that hill (a +6% pool setting, two 4-bit KV kernels, a tiered-cache patch) — the [history](docs/HISTORY.md) is mostly their graves.
2. **Trustworthy context capacity.** Agents live or die by how many deep sessions stay *warm*: a hit costs ~1–2 s on-GPU or ~2–7 s from the DRAM/NVMe tiers, where a cold 60K re-prefill costs ~11–13 s. So: the biggest KV pool that passes rule 1, then **~2.4M more tokens of it below the GPU** — and fp8 KV instead of denser-but-corrupting 4-bit kernels. Note the ordering: capacity that lies is worse than no capacity, which is why the tiers only shipped once they passed a cross-restart needle test and full-suite quality parity, not when their hit counters looked good.
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
- **~13.5K t/s prefill @8K** — native Blackwell FP4 GEMM; a cold 60K-token context loads in ~11–13 s on the tier profile.
- **200K usable context** on `fp8_e4m3` KV — the fp8 attention path is flat with depth (no decode crater) where custom 4-bit-KV kernels crater.
- **~2.59M reusable tokens across three KV tiers** — 214K on-GPU, ~245K in 24 GB of pinned DRAM, ~2.13M on 200 GB of NVMe. **The NVMe tier survives restarts**: a revisit after a container restart costs 4–7 s instead of a full re-prefill.
- **MTP speculative decoding at `ns=4`** — draft head inside the weights — crash-free under concurrency thanks to [PR #42603](https://github.com/vllm-project/vllm/pull/42603), and quality-neutral alongside the connector thanks to [two more local vLLM patches](patches/lmcache/README.md).
- **Vision** — the model's image tower, on.
- All of it on **one 32 GB RTX 5090** (`sm_120`), memory-OC'd, 600 W.

## Benchmarks

Hardware: RTX 5090 32 GB (`sm_120`, +4500 MHz mem OC, 600 W) + Ryzen 9 5900X + 64 GB RAM, Ubuntu 24.04.
Model: Qwen3.6-27B natfii NVFP4 W4A4 + `fp8_e4m3` KV + FlashInfer 0.6.15 + MTP `ns=4` + vision, `--no-async-scheduling`.
Tool: [llama-benchy](https://github.com/eugr/llama-benchy) 0.3.8. Full detail in [bench/RESULTS.md](bench/RESULTS.md).

> **Which profile these numbers are from.** The throughput matrices below were measured on the **no-LMCache** profile (util 0.98, pool 239,436, `mnbt` 4096) — they isolate the *engine*, and they're the numbers to compare against other setups. The tiers change capacity and revisit cost, not the decode/prefill rates: same weights, same kernels, same attention path. Where the tier profile differs (a ~10% narrower `mnbt` ceiling, a 25K smaller hot pool), it's quantified in [What removing LMCache changes](#what-removing-lmcache-changes) — read that before treating any row here as a tier-profile promise.

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

All numbers below are measured — the boot log (`gpu_worker` prints the breakdown at every launch) plus the checkpoint's safetensors headers for the weight split. Weights are identical in both profiles; the KV/util/sidecar rows are the daily's (tiers on, util 0.95):

| slice | GiB | notes |
|---|---:|---|
| **Weights, total** | **19.53** | split ↓ (19.15 on disk + loader overhead) |
| · language model | 12.76 | 64 hybrid layers (48 GDN + 16 full-attention), NVFP4 W4A4 + per-block scales |
| · embeddings | 2.37 | bf16 — not quantized |
| · lm_head | 2.37 | bf16 — embed+head = 4.7 GiB, a **24% big-vocab tax** on the weight budget |
| · vision tower | 0.86 | bf16, always loaded (even text-only requests) |
| · MTP drafter | 0.79 | shipped less-quantized than the previous daily's (0.28) — the price of the ns=4 head that ~2×'s deep decode |
| **KV pool** | **7.98** | **= 214,084 tokens** @ ~39 KiB/tok (fp8 attention KV + GDN state pages, one unified pool) |
| **LMCache sidecar** | **0.78** | 796 MiB of CUDA context + kernel modules, **invisible to `--gpu-memory-utilization`** — this is what caps the tier profile at util 0.95. Was 1,412 MiB before the staging-batch diet |
| **Peak-activation reserve** | 1.89 | sized by profiling at `mnbt` 3231 |
| **CUDA graphs + non-torch** | 0.40 | |
| **util 0.95 budget** | **29.78** | of 31.35 usable; killer-shape floor **531 MiB** free — healthier than the plain profile's 130–190 MiB at 0.98 |

Honest trade vs the previous daily: natfii's weights are **1.8 GiB heavier** in VRAM (fatter MTP head + FP4 scale tensors), which is why the hot pool is 214K where the old W4A16 daily reached 270K. We took it: prefill 3.4×, deep-concurrent throughput 2.2×, equal quality — and the tiers put ~2.4M tokens back underneath. Capacity you reload in 2–7 s is worth more than capacity you wait on.

What each token costs and what a cache hit is worth (60K-token context, measured):

| tier | capacity | revisit cost | survives restart |
|---|---|---|---|
| GPU pool (vLLM prefix cache) | 214,084 tok | **~1–2 s** (≈ decode time only) | no |
| host DRAM (LMCache L1, 24 GiB pinned) | ~245K tok @ ~98 KiB/tok serialized | **~2 s** | no |
| NVMe (LMCache L2, 200 GiB) | ~2.13M tok | **~4.4–7.5 s** | **yes** |
| miss → full re-prefill | — | **~11–13 s** (was ~23 s on the W4A16 daily — the FP4-GEMM dividend) | — |

**≈2.59M reusable tokens** total, of which ~2.13M persist across restarts.

Notes that save you from wrong conclusions:

- **util is the pool lever that matters** (+~8.4K tok per 0.01, measured across 0.92 → 0.98). `max-num-seqs` is *nearly* free but not exactly: `align` packs GDN state into the unified pool and consumes it per **active** request, so seqs 4 vs 8 measured identical — but seqs 8 → 16 costs **2,817 tokens (−1.3%)**, presumably per-slot bookkeeping crossing block granularity. Don't treat it as a capacity knob in either direction; do re-read the pool from the boot log if you change it, because a sweep run at a different `max-num-seqs` isn't strictly comparable to one that wasn't.
- The in-pool ~39 KiB/tok is an *effective average*: full-attention layers pay per-token fp8 KV; GDN layers pay a fixed per-sequence state that amortizes with depth. Serialized tiers pay ~98 KiB/tok because LMCache ships whole 1616-token unified blocks (attention KV + state page + metadata, padded to full block). **A tier token is ~2.5× the bytes of a pool token** — that ratio, not the tier's raw size, is what you budget against.
- Effective deep concurrency is **pool-bound, not `max-num-seqs`-bound**: 4 × ~50K-token agent sessions fill the hot pool; the fifth doesn't queue any more, though — it demotes to a tier and comes back in seconds. That's the whole point of the tiers.
- **The tiers required six local patches, none upstream.** Two of them are on **vLLM itself** and only fire with a connector + MTP + a hybrid model: a connector-path prefix-hit reduction that mixed an EAGLE-adjusted attention hit with an unadjusted Mamba hit (leaving one allocated-but-never-filled attention block at *every* local hit — ~10 eval points), and a store-boundary fix that stopped LMCache exporting vLLM's *null Mamba block* under a valid hash. Two more make LMCache's fp8 page regrouping correct at all; two are operational (sidecar VRAM, L2 cap enforcement). Table and failure modes: [patches/lmcache/README.md](patches/lmcache/README.md); the investigation: [docs/LMCACHE.md](docs/LMCACHE.md). Whatever you run: **needle-test across a restart before trusting any external KV tier on a hybrid model** — hit counters and coherent output do not prove fidelity. They looked perfect through four rounds of being wrong.

## What removing LMCache changes

[`scripts/serve-plain.sh`](scripts/serve-plain.sh) is the same engine with the connector and sidecar removed. It is a legitimate config — it's what every benchmark matrix above was measured on — and it is what you should run if you can't or won't carry six local patches.

| | **daily** (`serve.sh`, tiers on) | **plain** (`serve-plain.sh`) |
|---|---|---|
| GPU KV pool | 214,084 tok | **239,436 tok** (+25,352, +12%) |
| DRAM tier | ~245K tok (~2 s revisit) | — |
| NVMe tier | ~2.13M tok (~4.4–7.5 s, restart-proof) | — |
| **total reusable** | **~2.59M tok** | 239K tok |
| after a restart | ~2.13M tokens still warm | **everything cold** |
| `--gpu-memory-utilization` | 0.95 (sidecar takes 796 MiB it can't see) | **0.98** |
| `--max-num-batched-tokens` | 3231 (forced: LMCache needs 2·chunk−1) | **4096** (the deep-prefill optimum) |
| patches required | base 3 + **6 local LMCache/vLLM** | **base 3** |
| host resources | 24 GB pinned RAM + 200 GB SSD + a sidecar process | none |
| quality (69×2) | 89 | ~89.8 pooled (band 86–90) |

So the trade is: **give up 25K hot tokens and the `mnbt` 4096 prefill optimum; get ~2.4M tokens of second-chance capacity and a warm start after every restart.** Drop LMCache when:

- **your prompts are mostly fresh.** Tiers only pay off on *revisits*. One-shot chat, batch generation, or anything without long shared prefixes gets nothing back for the 25K tokens it gave up.
- **you can't run local patches.** Stock LMCache on this model is not a degraded version of this profile — it is silently wrong. `serve-plain.sh` is strictly better than an unpatched tier stack.
- **you don't have the host headroom.** 24 GB of *pinned* RAM is unswappable, and the L2 tier will use its full 200 GB cap — on the same filesystem your models and images live on.
- **you're chasing benchmark numbers.** The plain profile's larger pool and wider `mnbt` are worth a few percent on synthetic deep-prefill runs.

Keep LMCache when several agents share large prefixes, sessions are revisited across hours or restarts, or your working set exceeds the hot pool — the regime this box is actually for.

**The agentic A/B that settles it.** Same 16 SWE-Bench-Verified tasks ([R2E-Gym](https://github.com/R2E-Gym/R2E-Gym) / DeepSWE harness), 4 concurrent agents, 100-step cap, native tool calling, identical engine — the only difference is the connector:

| | tiers on | plain |
|---|---|---|
| wall-clock | **786 s** | 2,707 s (**3.4× slower**) |
| solved | 12 / 16 | 13 / 16 |
| avg steps | 46.2 | 47.7 |
| productive turns | 723 | 748 |
| prefix hit (in-GPU) | 46.2% | 8.7% |
| prefix hit (external) | **88.8%** | — |

An agent step resends its whole growing transcript, so nearly every request is a long prefix revisit — exactly the workload tiers are for. Without them, 91% of those prefixes miss and get re-prefilled; with them, nine in ten come back from DRAM or NVMe. **3.4× the throughput on identical work.** The one-task difference in solve rate (12 vs 13) is within noise at n=16 and runs the *opposite* direction to the throughput result — worth re-checking on a larger set before reading anything into it, but it's the honest number.

**How many agents? Four.** Concurrency sweep on the same harness (24 tasks per arm, tier profile at `--max-num-seqs 16` / pool 211,267 so c8 wasn't scheduler-throttled):

| agents | wall-clock | solved | avg queue depth | external prefix hit |
|---|---|---|---|---|
| **c4** | **1,388 s** | 19/24 | 0.4 | **89.5%** |
| c6 | 1,982 s (+43%) | 19/24 | 2.0 | 75.4% |
| c8 | 3,091 s (+123%) | 18/24 | 2.7 | 48.4% |

c4 already saturates the engine (the `mnbt 3231` prefill ceiling); past it, extra streams only queue and evict each other's prefixes out of the 24 GB L1 — the hit rate halves by c8 and wall-clock more than doubles for zero solve gain. This is the L1 sizing rule ("L1 must exceed hot-working-set ÷ 0.8") showing up as a throughput knee: more concurrent agents = bigger hot working set. If you need more than ~4 heavy agents, grow `--l1-size-gb` before growing concurrency.

**Official SWE-Bench-Verified score: 69.4% (347/500)** — the *complete* benchmark, single attempt per task, zero retries, patches replayed through the official `swebench` harness (0 evaluation errors). ~12 h of GPU time total at c4 on this one box, external prefix hit 78.7–84.6%. R2E-Gym's own reward signal said 349/500 — final divergence just 8 tasks in one direction and 6 in the other, so the harness reward is a faithful proxy for relative comparisons like the A/B and sweep tables above.

One methodology note, disclosed because it moved the number: R2E's task images ship locally-modified build files (`tox.ini`, `pyproject.toml`), and the exported `git diff` carries those image artifacts inside every patch — on the official checkout they don't apply, and swebench's `patch` fallback reverse-applies and breaks the tree (this mechanically zeroed all sphinx and most astropy tasks at first, reading 62.2%). We strip root-level build/config hunks (`tox.ini`, `pyproject.toml`, `setup.cfg`, `setup.py`) from patches that also touch source files — uniformly, all 500 tasks — and rescore. The agent's actual source edits are untouched.

Calibration against other Qwen3.6-27B numbers: 67.8% is the published same-model mini-swe-agent reference, 79.2% the public SOTA, 88–90% only with heavily engineered claude-cli agent stacks. A stock R2E scaffold on one RTX 5090 with tiered KV lands slightly above the reference band — the remaining headroom is agent-scaffold engineering, not engine configuration. Per-repo: django 167/231, sympy 52/75, sphinx 28/44, scikit-learn 28/32, matplotlib 20/34, astropy 11/22.

**Terminal-Bench 2.1: 48.3% (43/89)** — Harbor + terminus-2 (the leaderboard reference agent), official dataset, single attempt, default per-task timeouts (raising them is disqualifying: leaderboard validation requires `timeout_multiplier = 1.0`), this same tier profile. The anatomy is the honest part: only 17 of the 46 misses are genuine task failures — **27 are agent *timeouts*** (2 are harness errors). On the 60 tasks that finished within budget the pass rate is **71.7% (43/60)**. The timeouts are not a scheduling artifact: a c4-vs-c2 A/B reran every c4 timeout at double the per-stream budget and rescued zero — per-stream decode only moves ~72→85 tok/s median with concurrency (weight-bound W4A4 amortizes batching), and host CPU never passed ~19% (the benchmark itself caps most task containers at 1 CPU). What actually burns the clock is **token appetite at a hard ~130–140 tok/s per-stream ceiling**: the model writes 96–234K reasoning tokens on the hardest tasks, which no consumer-GPU decode rate fits inside a 900 s budget. So on this benchmark a single 5090 is wall-clock-bound before it is capability-bound — but buying score means materially faster decode (or shorter thinking), not concurrency tuning, and never timeout inflation.

**Operationally**, the tier profile also asks more of you: verify the pool is ~214K at boot (239K means the connector silently didn't attach), watch `du -sh` on the L2 directory for the first day, and wipe any L2 namespace written by a pre-patch build — poisoned chunks are not repaired.

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
# 1. base image (~1 min of pure-Python patches on top of the vLLM base;
#    the FlashInfer 0.6.15 step JIT-compiles its kernels on first run, so mount its cache)
cd patches && docker build -t vllm-qwen36:patched . && cd ..

# 2. tier image — LMCache main + the six patches, each gated by its own regression test
#    (this one compiles CUDA/C++, so it's minutes not seconds)
cd patches/lmcache && docker build -t vllm-qwen36:tiers . && cd ../..

# 3. serve the daily (tiers on)
./scripts/serve.sh
```

Then `http://localhost:8020/v1` speaks OpenAI. Every flag is explained inline in [`scripts/serve.sh`](scripts/serve.sh) and in [Config essentials](#config-essentials) below.

**Want it without the tiers?** Skip step 2 and run [`./scripts/serve-plain.sh`](scripts/serve-plain.sh) instead — bigger hot pool, no sidecar, base image only. [What you trade](#what-removing-lmcache-changes).

## What's in the patch stack

Three in the base image (below), plus [**six more for the KV tiers**](patches/lmcache/README.md) in the tier image — four on LMCache (fp8 page regrouping ×2, sidecar VRAM, L2 cap enforcement) and two on vLLM (connector-path EAGLE×hybrid hit reduction, Mamba store boundary). The base image also carries an (unused-by-this-config) TurboQuant 4-bit-KV stack documented separately in [docs/HISTORY.md](docs/HISTORY.md).

| patch | what it does |
|---|---|
| [`install_pr42603_sync.py`](patches/install_pr42603_sync.py) — [PR #42603](https://github.com/vllm-project/vllm/pull/42603) | **The fix that makes MTP `ns=4` usable on Blackwell.** One `torch.accelerator.current_stream().synchronize()` in the MTP draft loop, after the cudagraph-buffer writes and before the draft forward — closes the stale-buffer race that IMA-crashes `ns≥2` + fp8 KV under concurrency ([#40756](https://github.com/vllm-project/vllm/issues/40756), [#35288](https://github.com/vllm-project/vllm/issues/35288)). Perf-neutral. [Validated numbers](bench/RESULTS.md#mtp-k4-restored-on-blackwell--the-fp8-kv-spec-decode-crash-pr-42603). |
| FlashInfer 0.6.15 (Dockerfile pip step) | Latest FlashInfer, carrying the `sm_120` GDN/TMA fixes. cu130 AOT cubin/jit-cache isn't published for .15, so the image drops the mismatched 0.6.13 caches and lets 0.6.15 JIT-compile at runtime — **mount `/root/.cache/flashinfer`** (one build, warm forever). |
| [PR #44993 graft](https://github.com/vllm-project/vllm/pull/44993) — `v1/structured_output/__init__.py`, `v1/core/sched/scheduler.py` | **Structured output that survives thinking.** With a reasoning model, `response_format` json_schema + thinking-on returned EMPTY `content` (the schema JSON leaked into `reasoning_content`) — `should_advance`'s delta window skips `</think>` when MTP rejects drafts, so the grammar never re-engages. Needs `--structured-outputs-config '{"backend":"xgrammar","reasoning_parser":"qwen3","enable_in_reasoning":false}'`. Two pure-Python files. |

## Config essentials

`./scripts/serve.sh` runs the daily (flags also annotated inline there). The **tier-specific** ones first:

- `--kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'` — the **only** connector that handles this hybrid's opaque Mamba state ([LMCache PR #3613](https://github.com/LMCache/LMCache/pull/3613)). Needs the out-of-process `lmcache server` sidecar (ZMQ :5555) that owns the DRAM/NVMe tiers.
- `--max-num-batched-tokens 3231` = **2·chunk−1**, and `--chunk-size 1616` must equal vLLM's **unified block size** (1616 with MTP `ns=4`, 1568 without — discovered, not documented; it is not 16). Mismatch → "chunk size must be a multiple of vLLM block size". This ceiling is why the tier profile can't use the plain daily's `mnbt` 4096.
- `--gpu-memory-utilization 0.95` — **not 0.98.** The sidecar holds ~796 MiB of VRAM this flag cannot see. Validated by an 858-cycle soak; fallbacks 0.94 (205,633) and 0.92 (185,538).
- `-e LMCACHE_MP_GPU_STAGING_BATCH_SIZE=1 -e CUDA_MODULE_LOADING=LAZY` — the sidecar diet: 1,412 → 796 MiB at zero latency cost, which is exactly what bought util 0.95 over 0.92.
- `--l1-size-gb 24` (pinned host RAM; `drop_caches` first) — **must exceed hot-working-set / 0.8**, or an LRU head-chunk cascade drops the hit rate to **0%**. Partial caching does not degrade gracefully on this hybrid; undersize it and the cache is inert, not merely smaller.
- `--worker-reap-timeout-seconds 0` — reaper **off**. Default 120 s + a lazily-started heartbeat means one long idle span reaps the worker, after which the cache is an unrecoverable zombie (`found_count=0`, stores silently dropped).
- L2 `"eviction": {"eviction_policy":"LRU","trigger_watermark":0.8,"eviction_ratio":0.2}` — patch 0008 makes `max_capacity_gb` enforceable; **this block is what actually evicts**. You need both. Without them L2 grew to 876 GB against a 60 GB cap and filled the host root filesystem.
- **Never** `PYTORCH_ALLOC_CONF=expandable_segments` — cuMem/VMM memory is not CUDA-IPC-exportable ([pytorch#165685](https://github.com/pytorch/pytorch/issues/165685), [vllm#29544](https://github.com/vllm-project/vllm/issues/29544)); the sidecar can't import the KV handles and `register_kv_caches` silently times out at 300 s. The most expensive gotcha in this project — it read as version skew for days.
- `--ipc=host` + `--entrypoint bash` — CUDA-IPC needs host IPC, and the image entrypoint is `vllm serve`, which would swallow the `bash -c`.

And the ones shared with the plain profile:

- **No `--quantization` flag** — the natfii checkpoint is a ModelOpt NVFP4 export; vLLM auto-detects it and dispatches the CUTLASS FP4 GEMM kernels (`FlashInferCutlassNvFp4LinearKernel`). Forcing a flag here selects the wrong kernel path.
- `-e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728` — caps FlashInfer's lazily-allocated autotune workspace at 128 MiB; part of what makes util 0.98 survivable (gotcha #8). Perf-neutral in every regime we measured.
- `--kv-cache-dtype fp8_e4m3` + `-e VLLM_ATTENTION_BACKEND=FLASHINFER` — the flat-with-depth attention path; `e5m2` is **not** usable on this checkpoint (vLLM: "fp8_e5m2 not supported with fp8 checkpoints").
- `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":4}'` — MTP `ns=4`. **Requires the [PR #42603](https://github.com/vllm-project/vllm/pull/42603) graft** or it IMA-crashes under concurrency.
- **`--no-async-scheduling` — keep it.** MTP emits a multi-token verify batch every step; vLLM's async scheduler desyncs its request-ID→batch-row mapping under spec decode ([vllm#42655](https://github.com/vllm-project/vllm/issues/42655)) and corrupts KV. Async-off is the documented fix.
- `--mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill` — hybrid-model prefix caching; **`align` is load-bearing for the pool size** (it packs the GDN/Mamba state into the unified KV pool) *and* for the tiers (it's what gives LMCache a scheduler-sized page to store). Mode `all` is same speed / same pool and does **not** avoid the crash — don't bother.
- `--max-model-len 200000`, and let vLLM profile the pool — don't hand-set `--kv-cache-memory` (its "fully utilize" hint ignores warmup transients and OOMs). util is the only pool lever (~+8.4K tok/0.01); `mnbt` doesn't change the pool. **The util ceiling is model-specific, not a constant**, and the sidecar moves it: the plain profile runs **0.98 → 239,436** (fallback 0.96 → 222,535), the tier profile **0.95 → 214,084** (fallbacks 0.94 → 205,633, 0.92 → 185,538). 0.98 killed the previous, heavier-shaped daily at serve time; re-validate against gotcha #8's killer shape if you change anything.
- `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml` — `qwen3_xml` is correct; `hermes` silently drops tool calls.
- `--override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}'` + `--default-chat-template-kwargs '{"preserve_thinking":true}'` — keep historical `<think>` blocks across turns. **Caveat:** the *client* must resend prior reasoning in the **`reasoning`** field (not the deprecated `reasoning_content`, which vLLM ignores on input).
- `--structured-outputs-config '{"backend":"xgrammar","reasoning_parser":"qwen3","enable_in_reasoning":false}'` — enables the reasoning gate the [#44993](https://github.com/vllm-project/vllm/pull/44993) graft needs. Give it an adequate `max_tokens` budget (reasoning + JSON) or it truncates mid-think and looks empty.
- `--limit-mm-per-prompt '{"image":4,"video":0}'` — vision on.

**Verify container identity after launch.** Confirm the startup log reports the pool you expect: **~214K on the tier daily**, **~239K on the plain profile**. On the tier daily, seeing 239K is the specific tell that **the connector didn't attach** — you booted the plain engine by accident and every "tier hit" you go on to measure will be vLLM's own prefix cache. That exact mistake cost us four rounds of analysis; it reads as a perfectly healthy server the whole time.

## Gotchas that bite during setup

1. **MTP `ns≥2` needs [PR #42603](https://github.com/vllm-project/vllm/pull/42603) or it IMA-crashes under concurrency.** Single-stream and `ns=1` pass every test and hide it; `CUDA_LAUNCH_BLOCKING=1` masks it. **Load-test with 3+ parallel streams on day one** — that's the only thing that reproduces it.
2. **`--no-async-scheduling` is mandatory with MTP.** Async scheduling desyncs the request-ID→batch-row mapping under spec decode ([#42655](https://github.com/vllm-project/vllm/issues/42655)) and corrupts KV.
3. **Verify the KV pool in the launch log** (~214K tiers / ~239K plain). A silent config fallback (wrong preset, `align` off, wrong image, connector not attached) looks like a healthy server at the wrong pool.
4. **flashinfer JIT eats all host RAM (non-nightly images).** Any non-nightly vLLM image on `sm_120` JIT-compiles CUTLASS kernels on the first forward with unbounded `nvcc` parallelism — multi-GB per job, reads as a mystery "hang" or whole-host livelock. Cap it (`MAX_JOBS=4` + `FLASHINFER_NUM_COMPILE_JOBS=4`) and **mount a persistent `/root/.cache/flashinfer`** (one build, warm forever).
5. **vLLM's prefix-cache metric lies on this model.** `vllm:prefix_cache_hits_total` / "Prefix cache hit rate: 0.0%" report **0% while the cache works**. Don't debug the counter — time a repeated prompt ([`bench/prefix_probe.py`](bench/prefix_probe.py)). (With the connector on, "External prefix cache hit rate" *is* meaningful and reports the tier hits — but see gotcha #10: a hit rate proves a *lookup* matched, never that the bytes were right.)
6. **Validate coherence via raw `/v1/completions`.** The chat endpoint's reasoning parser swallows degenerate output as *empty content*, so a broken model looks "fine but quiet." Tells: constant-token output, or **flat 100% MTP acceptance** (draft and verify locked in step).
7. **Re-verify after every vLLM bump.** These grafts are version-sensitive; a nightly that moves `llm_base_proposer.py` will fail the `ast` check at build (loud) or, worse, shift the anchor.
8. **The util ceiling is set by *lazy autotune workspace*, boot-margin probes cannot see it, and it is model-specific.** The first time the engine meets a genuinely new batch shape (e.g. a fresh 8×8K concurrent prefill+decode wave), the fp4-GEMM/FlashInfer autotuner allocates its benchmark workspace **at serve time**: measured ~266 MiB for `mnbt` 4096 shapes, ~486 MiB for 8192 shapes, on top of allocator fragmentation. This OOM-killed the previous (W4A16) daily at util 0.98 mid-traffic, 100% reproducibly (`benchy --pp 8192 --concurrency 8` is the reliable trigger) — even though near-pool-full text bursts and 8×4-image vision bursts all passed. The **plain** profile *does* run 0.98, and earned it (the tier profile sits at 0.95 because the sidecar eats the margin): `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728` caps the workspace, both serve scripts pre-warm the killer shape at boot so the allocation happens before real traffic, and the config passed a 0.98 battery ending in two *simultaneous* waves of 8 deep-text floods + 8 four-image vision requests + `pp8192×c8` benchy on a cold engine (steady-state floor: ~130 MiB free, stable across waves). Traps when you probe a ceiling yourself: (a) identical prompts are silently collapsed by prefix caching — use distinct prompts; (b) text-only bursts miss the vision-encoder transient; (c) **prefill-style bursts miss the autotune transient — include a fresh deep-prefill × full-decode shape (`pp8192 × c8`), and fire your stressors *concurrently*, not sequentially.** `mnbt` 8192 needs more margin (≲0.94) — its bigger shapes want the bigger workspace.
9. **Check `tokenizer.json` for baked truncation.** This checkpoint shipped with `"truncation": {"max_length": 8192}` left over from calibration — a silent poison pill: text works, single small images work, but any multimodal request expanding past 8192 tokens hard-fails with a processor mismatch (HTTP 400). Both serve scripts verify and null it at every launch, because **re-downloading the model reintroduces the bug**.

And three more that only bite once the KV tiers are on:

10. **An unpatched external KV tier on a hybrid model is worse than no tier.** It stores wrong-addressed pages and restores garbage recurrent state — with *fluent output, rising hit counters, and zero errors logged*. The only test that catches it is a **needle planted in a long context, retrieved after a restart**. We shipped "validated" tier profiles twice before running that test; both were storing nothing or storing garbage. Run it first, run it after every bump.
11. **`max_capacity_gb` on the `fs_native` L2 adapter is telemetry, not a limit** (unpatched). Per-adapter eviction is also opt-in. Ours grew to **876 GB against a 60 GB cap** and filled the host root filesystem overnight. Patch 0008 adds reserve-before-write admission and the `"eviction"` block does the evicting — but whatever you run, **`du -sh` the L2 directory on a timer for the first day**. An unenforced cache cap on your root disk is a self-brick timer.
12. **The sidecar's VRAM is invisible to `--gpu-memory-utilization`.** ~796 MiB after the staging diet, 1,412 MiB before it. If you port this profile and it OOMs at a util that "worked", this is why — it's also the entire origin of the folk belief that "MTP crashes with LMCache". It doesn't; it runs out of memory.

## Daily lineage — what each daily was, and why the next took over

Newest first. Every switch is documented with numbers in [docs/HISTORY.md](docs/HISTORY.md) and [bench/RESULTS.md](bench/RESULTS.md).

| daily | weights · KV | pool | why it took over |
|---|---|---|---|
| **+ LMCache DRAM/NVMe tiers (current, 2026-07-20)** | *(same engine)* | **214K** @0.95 **+ ~2.4M tiered** | Six local patches made tiered KV *faithful* on this fp8 hybrid (cross-restart needle + 69×2 = **89** vs a ~89.8 baseline). Trades 25K hot tokens and `mnbt` 4096 for ~2.4M tokens of second-chance capacity and a **warm start after restarts** — a 60K revisit costs 2 s (DRAM) or 4–7 s (NVMe) instead of an 11–13 s re-prefill. Validated by an 858-cycle soak (flat VRAM, L2 stable under its cap). [`serve-plain.sh`](scripts/serve-plain.sh) keeps the row below available. |
| natfii NVFP4 W4A4 · fp8_e4m3 + FlashInfer + MTP `ns=4` (2026-07-19) | NVFP4 W4A4 · fp8 | ~239K @0.98 | **Prefill 3.4×** (13.5K vs 4.0K t/s @8K — native Blackwell FP4 GEMM vs Marlin dequant), deep-concurrent sustained **2.2×** (148 vs 67 t/s at pp30K×c8 tg512), cold 60K context 10 s vs 23 s — at **equal 69×2 quality** (~90, 4 trials each side; the W4A4 activation cost was bounded at ≈1 pt via a chimera A/B and natfii's calibration covers it). Survived the full promotion gauntlet incl. a 106-cycle soak and a 0.98 combined-wave battery. Pool is 11% smaller than AR's 270K (heavier MTP head + FP4 scales) — traded for re-prefilling 3× faster. |
| Lorbus INT4-AutoRound · fp8_e4m3 + FlashInfer + MTP `ns=4` (2026-07-18) | INT4-AutoRound · fp8 | ~270K @0.96 | Flat deep decode (fp8+FlashInfer has **no** decode crater at depth, where the custom TurboQuant kernel drops); biggest pool ever; **MTP `ns=4`** restored by [PR #42603](https://github.com/vllm-project/vllm/pull/42603); tool-eval 90; dropped the experimental TurboQuant KV kernel for the **battle-tested fp8** path. (A one-day 0.98/287K promotion was reverted the same night: serve-time autotune OOM — gotcha #8.) |
| turboquant_4bit_nc (NVFP4) + MTP `ns=3` (2026-07-15) | NVFP4 · TQ 4-bit K/V | ~235K | +42% pool over k8v4, once the "4bit_nc destroys retrieval" **0/8** was traced to the async×spec KV confound and fixed with `--no-async-scheduling`. Decode still craters at deep single-stream (the custom-kernel cost). |
| turboquant_k8v4 (NVFP4) | NVFP4 · TQ 8-bit K/4-bit V | ~165K | +21% pool over fp8 at fp8-equal retrieval quality (8-bit keys). |
| fp8_e4m3 (stock nightly) | NVFP4 · fp8 | ~136K | The original battle-tested baseline — flat deep decode, no patches, smallest pool. |

The arc, compressed: the fp8 baseline's **flat-decode virtue** survived every generation; PR #42603 added working `ns=4`; AutoRound added quality and pool; NVFP4 W4A4 cashed in the GPU's native FP4 compute — the first daily where *prefill* got a generational jump instead of decode or capacity; and the KV tiers finally broke capacity out of the 32 GB box entirely, which is the one axis a bigger GPU would otherwise have been the only answer to.

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
