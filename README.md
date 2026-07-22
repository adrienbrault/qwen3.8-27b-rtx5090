# Qwen3.6-27B on a single RTX 5090 — SWE-Bench Verified 69.4%, W4A4 NVFP4, 13.5K t/s prefill, ~2.6M tokens of tiered KV

Serving **Qwen3.6-27B at 200K context with ~13.5K t/s prefill, MTP speculative decoding at `ns=4`, vision, and a three-tier KV cache — 214K tokens on-GPU, ~245K in pinned DRAM, ~2.13M on NVMe** — on one 32 GB consumer GPU (RTX 5090, Blackwell `sm_120`). The NVMe tier survives restarts, so yesterday's agent session is still warm this morning.

> **Validated end to end, not just on throughput**: this exact profile scores **69.4% on SWE-Bench-Verified** (full 500, single attempt, official `swebench` harness) and **48.3% on Terminal-Bench 2.1** (Harbor + terminus-2, default timeouts) — agents hammering the box for ~12 h stretches, zero engine crashes. [The numbers and their anatomy →](#agentic-benchmark-results)

The daily is the **[natfii NVFP4 W4A4](https://huggingface.co/natfii/Qwen3.6-27B-VLM-NVFP4-MTP)** checkpoint + **`fp8_e4m3` KV cache** + **FlashInfer** attention + **MTP `ns=4`** + **[LMCache](https://github.com/LMCache/LMCache) DRAM/NVMe offload**, on a patched vLLM image. Two things carry it:

- **W4A4 turns on Blackwell's native FP4 tensor cores** — **3.4× the prefill** of the weight-only-quant daily it replaced, at equal measured quality ([why, in DESIGN.md](docs/DESIGN.md#why-these-weights--and-what-actually-governs-prefill-speed)). The patch that makes `ns=4` possible at all on Blackwell is [vLLM PR #42603](https://github.com/vllm-project/vllm/pull/42603); without it, MTP + fp8 KV illegal-memory-access-crashes under any real concurrency.
- **The KV tiers turn eviction into a 2–7 s reload instead of an 11–13 s re-prefill**, and turn a restart from a total cache loss into a warm start. On a 16-task SWE-Bench-Verified run at 4 concurrent agents that's **3.4× the wall-clock throughput** ([A/B](docs/LMCACHE.md#what-removing-lmcache-changes)) — agents resend their whole transcript every step, so almost every request is a long prefix revisit. Getting them *faithful* on an fp8 hybrid took **[six local patches](patches/lmcache/README.md)** — four on LMCache, two on vLLM. Unpatched, this profile is worse than no cache at all: it stores wrong-addressed pages and restores garbage recurrent state, with fluent output and zero errors logged.

If you want the engine without the tiers — bigger hot pool, no sidecar, no local LMCache patches — that's [`scripts/serve-plain.sh`](scripts/serve-plain.sh), and the exact trade is spelled out in [What removing LMCache changes](docs/LMCACHE.md#what-removing-lmcache-changes).

## The config at a glance

| | |
|---|---|
| model | [natfii Qwen3.6-27B-VLM-NVFP4-MTP](https://huggingface.co/natfii/Qwen3.6-27B-VLM-NVFP4-MTP) — W4A4, MTP head, vision tower |
| engine | patched vLLM (3 base + 6 tier patches) + FlashInfer 0.6.15, `fp8_e4m3` KV, MTP `ns=4`, `--no-async-scheduling` |
| context / hot pool | 200K / **214,084 tokens** on-GPU (util 0.95; plain profile: 239,436 @0.98) |
| tiered KV | + ~245K tok pinned DRAM (~2 s revisit) + ~2.13M tok NVMe (~4.4–7.5 s, **survives restarts**) ≈ **2.59M reusable** |
| prefill | **~13.5K t/s @8K** (native Blackwell FP4 GEMM); cold 60K context ~11–13 s |
| decode | **~80–160 t/s single-stream, content-dependent** (MTP acceptance: creative prose ~82, code ~158; benchy ~116–140); flat with depth to 180K; aggregate peaks 700–930 t/s (c8, warm) |
| quality | tool-eval-bench **~90**/100 (full 69×2, ×4 runs) — parity with the best W4A16 daily |
| **SWE-Bench-Verified** | **69.4%** (347/500, official harness, single attempt) |
| **Terminal-Bench 2.1** | **48.3%** (43/89, terminus-2, default timeouts; **71.7%** on tasks that finished within budget) |
| hardware | 1× RTX 5090 32 GB (+4500 MHz mem OC, 600 W) + Ryzen 9 5900X + 64 GB RAM |
| endpoint | OpenAI-compatible, `http://localhost:8020/v1` |

## What this config optimizes for

This is a **daily driver for agentic coding** — a handful of coding agents with deep (8K–100K+) contexts, plus interactive chat and the occasional image, on one always-on box. That workload ranks the goals, and the ranking explains every choice below:

1. **Reliability over everything.** An engine that crashes mid-run or — worse — answers *fluently but wrongly* from corrupted cache is worth less than a slower one. Every config here survived a promotion gauntlet: concurrent burst battery, a fresh-deep-batch OOM trigger, needle-in-haystack recall across cache boundaries, and a 69-scenario × 2 tool-eval. Several faster configs died on that hill (a +6% pool setting, two 4-bit KV kernels, a tiered-cache patch) — the [history](docs/HISTORY.md) is mostly their graves.
2. **Trustworthy context capacity.** Agents live or die by how many deep sessions stay *warm*: a hit costs ~1–2 s on-GPU or ~2–7 s from the DRAM/NVMe tiers, where a cold 60K re-prefill costs ~11–13 s. So: the biggest KV pool that passes rule 1, then **~2.4M more tokens of it below the GPU** — and fp8 KV instead of denser-but-corrupting 4-bit kernels. Note the ordering: capacity that lies is worse than no capacity, which is why the tiers only shipped once they passed a cross-restart needle test and full-suite quality parity, not when their hit counters looked good.
3. **Latency in the agent regime, not benchmark aggregate.** For agents, latency *is* mostly prefill: every fresh deep context pays it up front, and under concurrency everyone queues behind it. W4A4 tripling the prefill lane is the single biggest felt improvement in this config's history. MTP `ns=4` then roughly doubles deep single-stream decode (the "agent reading its own long context" case) even though it does nothing for shallow batch throughput.
4. **Everything on at once.** Vision, 200K context, speculative decoding, reasoning + structured outputs, tool calling — the daily runs the full stack simultaneously. No per-benchmark specialization; the numbers below are the config you'd actually run.

Non-goals: maximum batched throughput for many shallow users (a serving-farm concern — this box peaks at ~500–800 t/s aggregate anyway when streams are warm), multi-GPU, and minimum VRAM.

## Benchmarks

Hardware: RTX 5090 32 GB (`sm_120`, +4500 MHz mem OC, 600 W) + Ryzen 9 5900X + 64 GB RAM, Ubuntu 24.04.
Model: Qwen3.6-27B natfii NVFP4 W4A4 + `fp8_e4m3` KV + FlashInfer 0.6.15 + MTP `ns=4` + vision, `--no-async-scheduling`.
Tool: [llama-benchy](https://github.com/eugr/llama-benchy) 0.3.8. Full detail and protocols in [bench/RESULTS.md](bench/RESULTS.md).

> **Which profile these numbers are from.** The throughput matrices below were measured on the **no-LMCache** profile (util 0.98, pool 239,436, `mnbt` 4096) — they isolate the *engine*, and they're the numbers to compare against other setups. The tiers change capacity and revisit cost, not the decode/prefill rates. Where the tier profile differs, it's quantified in [What removing LMCache changes](docs/LMCACHE.md#what-removing-lmcache-changes).

**Stability — the promotion gauntlet.** Zero-crash across: the ceiling battery at util 0.98 (needle-in-haystack, `pp8192×c8` + `pp30000×c8` killer shapes, 8× distinct ~34K text floods, 8× four-image vision bursts), two *simultaneous* combined waves (16 mixed requests + benchy on a cold engine), a **106-cycle overnight soak** (zero VRAM drift), and 4 full **69×2 tool-evals** under load.

**Decode — `t/s (total)` (aggregate), `--pp 512 4096 --tg 128 --concurrency 1 2 4 8 --runs 3`:**

| decode t/s (total) | c1 | c2 | c4 | c8 | c16 |
|---|---|---|---|---|---|
| pp512 | 116 | 213 | 358 | **706** (peak 933) | 593 (peak 898) |
| pp4096 | 126 | 204 | 280 | 352 (peak 854) | — |

> **Decode speed is content-dependent — honesty note.** With MTP speculative decoding, single-stream rate tracks how *predictable* the output is: the draft head lands more tokens per verify step on structured text. Measured on the daily, same 600-token budget: **"write a short story" → 82 t/s; "create a todo app" → 158 t/s.** The benchy rows above (~116–140) sit in the middle of that spread. Nothing in the tables is wrong — but if your workload is creative prose, read the low end; if it's code (this box's actual job), read the high end.

**Sustained at depth** (`tg 512`, aggregate t/s, peak during warm overlap in parens) — batched decode is fast on this hybrid; what drags deep cold-context numbers is the shared prefill lane, now 3.4× wider:

| sustained t/s (total) | c1 | c4 | c8 |
|---|---|---|---|
| pp512 | 125 | 422 (512) | **778 (961)** |
| pp4096 | 127 | 369 (495) | 605 (950) |
| pp8192 | 114 | 308 (533) | 466 (925) |
| pp30000 | 125 | 164 (481) | 149 (582) |
| pp90000 | — | 39 (263) | — |

**Prefill under concurrency** — saturated at c1, so aggregate stays flat and per-request divides by N; a cold 30K prefill occupies ~3.0 s of the lane (was ~8.6 s pre-W4A4):

| prefill t/s | c1 | c4 aggregate (per-req) | c8 aggregate (per-req) |
|---|---|---|---|
| pp8192 | 13,315 | 13,577 (~3,340) | 13,347 (~1,670) |
| pp30000 | 10,117 | 10,001 (~2,500) | 9,878 (~1,235) |
| pp90000 | — | 5,288 (~1,320) | — |

**Long context (c1)** — decode is **flat ~136–140 t/s from 30K → 180K** (no deep-context crater); TTFT is where W4A4 lands hardest:

| context | prefill t/s | e2e TTFT | decode t/s |
|---|---|---|---|
| 30K | 10,167 | **2.7 s** (was 7.4) | 136 |
| 90K | 5,780 | **14.1 s** (was 27.9) | 140 |
| 180K | 3,472 | **47.0 s** (was 72.4) | 138 |

**Quality — [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench): ~90**/100 (full 69-scenario suite × 2 trials, ×4 independent runs, pooled 89.8) — statistically indistinguishable from the best W4A16 daily (87.8 pooled, same protocol).

### Agentic benchmark results

Two official end-to-end benchmarks, run on this exact daily profile (tiers on), agents talking to the box like any other OpenAI endpoint:

| benchmark | score | harness / agent | shape |
|---|---|---|---|
| **SWE-Bench-Verified** | **69.4%** (347/500) | official `swebench` harness, R2E-Gym scaffold | full 500, single attempt, zero retries |
| **Terminal-Bench 2.1** | **48.3%** (43/89) | Harbor + terminus-2 (leaderboard reference agent) | k=1, default per-task timeouts |

**SWE-Bench-Verified 69.4%** lands above the published same-model mini-swe-agent reference (67.8%), under the 79.2% public SOTA — the remaining headroom is agent-scaffold engineering, not engine configuration. One methodology disclosure: exported patches need R2E image build-file hunks stripped before official replay, applied uniformly across all 500 tasks.

**Terminal-Bench 2.1 48.3%** is wall-clock-bound before it is capability-bound: 27 of the 46 misses are agent *timeouts* (only 17 genuine fails), and the pass rate on tasks that finished within budget is **71.7% (43/60)**. The clock goes to 96–234K-token reasoning traces against a ~130–140 t/s per-stream ceiling — concurrency tuning doesn't move it (measured), and raising timeouts would disqualify the number. For scale: terminus-2 leaderboard rows (k=5) run Fable 5 80.4%, Opus 4.7 66.1%; best open-weight row GLM-5.1 58.7%.

Full anatomy — the timeout A/B, the patch-sanitization mechanics, per-repo splits, leaderboard rules: [bench/RESULTS.md](bench/RESULTS.md#agentic-benchmarks--full-anatomy-2026-07-20--22-tier-daily).

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

Then `http://localhost:8020/v1` speaks OpenAI. Every flag is explained inline in [`scripts/serve.sh`](scripts/serve.sh) and in [docs/GOTCHAS.md](docs/GOTCHAS.md).

**Want it without the tiers?** Skip step 2 and run [`./scripts/serve-plain.sh`](scripts/serve-plain.sh) instead — bigger hot pool, no sidecar, base image only. [What you trade](docs/LMCACHE.md#what-removing-lmcache-changes).

## Before you run this — the three landmines

1. **MTP `ns≥2` + fp8 KV IMA-crashes stock vLLM under concurrency** — the image's [PR #42603](https://github.com/vllm-project/vllm/pull/42603) graft is the fix. Single-stream tests hide it; load-test with 3+ parallel streams on day one.
2. **`--no-async-scheduling` is mandatory with MTP** — vLLM's async scheduler corrupts KV under spec decode ([#42655](https://github.com/vllm-project/vllm/issues/42655)).
3. **Verify the KV pool in the launch log**: ~214K on the tier daily, ~239K plain. On the tier daily, 239K means the connector silently didn't attach — the server looks perfectly healthy while every "tier hit" you measure is vLLM's own prefix cache.

Nine more, plus three tier-specific ones (the 876 GB disk-fill, the invisible sidecar VRAM, the needle-test discipline): **[docs/GOTCHAS.md](docs/GOTCHAS.md)**.

## Why it needs a patch: MTP × fp8-KV × Blackwell crashes on stock vLLM

> ⚠️ **`ns≥2` MTP with `fp8_e4m3` KV on `sm_120` is a 100%-reproducible illegal-memory-access under concurrency** — a known, still-open upstream bug ([vllm#40756](https://github.com/vllm-project/vllm/issues/40756), same Qwen3.6-27B model; [vllm#35288](https://github.com/vllm-project/vllm/issues/35288)). It crashes at `rejection_sampler.py:267 parse_output → cudaErrorIllegalAddress`. Single-stream and `ns=1` are both clean; `CUDA_LAUNCH_BLOCKING=1` masks it (→ a timing race).

The root cause ([PR #42603](https://github.com/vllm-project/vllm/pull/42603)): the MTP draft loop writes shared cudagraph input buffers, then immediately launches the draft-model forward that reads them — **without a stream sync**. The fix is one line:

```python
self.input_ids[:batch_size] = input_ids
self.hidden_states[:batch_size] = hidden_states
torch.accelerator.current_stream().synchronize()   # PR #42603
```

Perf-neutral, validated crash-free across everything in [Benchmarks](#benchmarks). The multi-day bisection that found it — including every plausible fix that *didn't* work — is in [docs/HISTORY.md](docs/HISTORY.md).

## What's in the patch stack

Three in the base image (below), plus [**six more for the KV tiers**](patches/lmcache/README.md) in the tier image — four on LMCache (fp8 page regrouping ×2, sidecar VRAM, L2 cap enforcement) and two on vLLM (connector-path EAGLE×hybrid hit reduction, Mamba store boundary).

| patch | what it does |
|---|---|
| [`install_pr42603_sync.py`](patches/install_pr42603_sync.py) — [PR #42603](https://github.com/vllm-project/vllm/pull/42603) | **The fix that makes MTP `ns=4` usable on Blackwell.** One stream-sync in the MTP draft loop; closes the stale-buffer race above. Perf-neutral. [Validated numbers](bench/RESULTS.md). |
| FlashInfer 0.6.15 (Dockerfile pip step) | Latest FlashInfer, carrying the `sm_120` GDN/TMA fixes. cu130 AOT cubins aren't published for .15, so it JIT-compiles at runtime — **mount `/root/.cache/flashinfer`** (one build, warm forever). |
| [PR #44993 graft](https://github.com/vllm-project/vllm/pull/44993) | **Structured output that survives thinking.** With a reasoning model, `response_format` json_schema + thinking-on returned EMPTY `content`. Needs `--structured-outputs-config` (see [GOTCHAS.md](docs/GOTCHAS.md)). Two pure-Python files. |

## How we got here / what didn't work

The docs, by the question you're asking:

- **"Why these weights / where does the VRAM go / what does the host contribute?"** → [docs/DESIGN.md](docs/DESIGN.md) — the W4A4-vs-W4A16 prefill mechanics, the measured 31.35 GiB memory budget, tier cost per token, the OC.
- **"What do the KV tiers buy, and what breaks without the patches?"** → [docs/LMCACHE.md](docs/LMCACHE.md) — the trade table, the 3.4× agentic A/B, the c4 concurrency sweep, and the four rounds of silently-wrong "validated" profiles.
- **"Why doesn't my port of this config work?"** → [docs/GOTCHAS.md](docs/GOTCHAS.md) — every flag with its failure mode, plus the 12 setup gotchas.
- **"How did the config evolve / what's the full bug story?"** → [docs/HISTORY.md](docs/HISTORY.md) — the daily lineage table, the two reversals, the PR #42603 bisection.
- **"Has X been tried?"** → [docs/REJECTED.md](docs/REJECTED.md) — everything rejected, with the number that killed it. Read it before "improving" the config.
- **All raw numbers and protocols** → [bench/RESULTS.md](bench/RESULTS.md); probes in [bench/](bench/README.md).

## License

MIT (see [LICENSE](LICENSE)) for the original work here — docs, benchmarks, scripts, and the grafts. The vLLM files the patches modify stay under **Apache-2.0**; see [THIRD_PARTY.md](THIRD_PARTY.md).

## Credits

- [vLLM](https://github.com/vllm-project/vllm) and [PR #42603](https://github.com/vllm-project/vllm/pull/42603) — the draft-loop sync that makes MTP `ns=4` usable on Blackwell.
- [natfii](https://huggingface.co/natfii/Qwen3.6-27B-VLM-NVFP4-MTP) for the Qwen3.6-27B W4A4 NVFP4 export (ModelOpt 0.43) — the current daily's weights.
- [Lorbus](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) for the Qwen3.6-27B INT4-AutoRound quant — the previous daily, still the W4A16 reference.

---

<sub>Sections that used to live in this README, for old links:</sub>

#### What you get

Now [The config at a glance](#the-config-at-a-glance).

#### Why these weights — and what actually governs prefill speed

Moved to [docs/DESIGN.md](docs/DESIGN.md#why-these-weights--and-what-actually-governs-prefill-speed).

#### Where the 31.35 GiB goes — memory budget

Moved to [docs/DESIGN.md](docs/DESIGN.md#where-the-3135-gib-goes--memory-budget).

#### What removing LMCache changes

Moved to [docs/LMCACHE.md](docs/LMCACHE.md#what-removing-lmcache-changes).

#### Config essentials

Moved to [docs/GOTCHAS.md](docs/GOTCHAS.md#config-essentials).

#### Gotchas that bite during setup

Moved to [docs/GOTCHAS.md](docs/GOTCHAS.md#gotchas-that-bite-during-setup).

#### Daily lineage — what each daily was, and why the next took over

Moved to [docs/HISTORY.md](docs/HISTORY.md#daily-lineage--what-each-daily-was-and-why-the-next-took-over).
