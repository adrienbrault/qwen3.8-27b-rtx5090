# Qwen3.6-27B on a single RTX 5090 — SWE-Bench Verified 69.4%, W4A4 NVFP4, 13.5K t/s prefill, ~2.6M tokens of tiered KV

Serving **Qwen3.6-27B at 200K context with ~13.5K t/s prefill, MTP speculative decoding at `ns=4`, vision, and a three-tier KV cache — 214K tokens on-GPU, ~245K in pinned DRAM, ~2.13M on NVMe** — on one 32 GB consumer GPU (RTX 5090, Blackwell `sm_120`). The NVMe tier survives restarts, so yesterday's agent session is still warm this morning.

> **Validated end to end, not just on throughput**: this exact profile scores **69.4% on SWE-Bench-Verified** (full 500, single attempt, official `swebench` harness) and **48.3% on Terminal-Bench 2.1** (Harbor + terminus-2, default timeouts) — agents hammering the box for ~12 h stretches, zero engine crashes. [The numbers and their anatomy →](#agentic-benchmark-results)

The daily is the **[natfii NVFP4 W4A4](https://huggingface.co/natfii/Qwen3.6-27B-VLM-NVFP4-MTP)** checkpoint + **`fp8_e4m3` KV cache** + **FlashInfer** attention + **MTP `ns=4`** + **[LMCache](https://github.com/LMCache/LMCache) DRAM/NVMe offload**, on a patched vLLM image. Two things carry it:

- **W4A4 turns on Blackwell's native FP4 tensor cores** — **3.4× the prefill** of the weight-only-quant daily it replaced, at equal measured quality ([why, in DESIGN.md](docs/DESIGN.md#why-these-weights--and-what-actually-governs-prefill-speed)). MTP `ns=4` needs a one-line synchronization workaround (from [vLLM PR #42603](https://github.com/vllm-project/vllm/pull/42603), closed unmerged upstream) — without it, MTP + fp8 KV illegal-memory-access-crashes under any real concurrency.
- **The KV tiers turn eviction into a 2–7 s reload instead of an 11–13 s re-prefill**, and turn a restart from a total cache loss into a warm start. On a 16-task SWE-Bench-Verified run at 4 concurrent agents that's **3.4× the wall-clock throughput** ([A/B](docs/LMCACHE.md#what-removing-lmcache-changes)) — agents resend their whole transcript every step, so almost every request is a long prefix revisit. Getting them *faithful* on an fp8 hybrid took **[six local patches](patches/lmcache/README.md)** — four on LMCache, two on vLLM. Unpatched, this profile is worse than no cache at all: it stores wrong-addressed pages and restores garbage recurrent state, with fluent output and zero errors logged.

If you want the engine without the tiers — bigger hot pool, no sidecar, no local LMCache patches — that's [`scripts/serve-plain.sh`](scripts/serve-plain.sh), and the exact trade is spelled out in [What removing LMCache changes](docs/LMCACHE.md#what-removing-lmcache-changes).

## The config at a glance

| | |
|---|---|
| model | [natfii Qwen3.6-27B-VLM-NVFP4-MTP](https://huggingface.co/natfii/Qwen3.6-27B-VLM-NVFP4-MTP) — W4A4, MTP head, vision tower |
| engine | patched vLLM (3 base + 6 tier patches) + FlashInfer 0.6.15, `fp8_e4m3` KV, MTP `ns=4`, `--no-async-scheduling` |
| context / hot pool | 200K / **214,084 tokens** on-GPU (util 0.95; plain profile: 239,436 @0.98) |
| tiered KV | + ~245K tok pinned DRAM (~2 s revisit) + ~2.13M tok NVMe (~4.4–7.5 s, **survives restarts**) ≈ **2.59M reusable** — aggregate *reusable prefixes* across sessions, **not** the per-request window (that caps at 200K) |
| prefill | **~13.5K t/s @8K** (native Blackwell FP4 GEMM); cold 60K context ~11–13 s |
| decode | **~80–160 t/s single-stream, content-dependent** (MTP acceptance: creative prose ~82, code ~158; benchy ~116–140); flat with depth to 180K; aggregate peaks 700–930 t/s (c8, warm) |
| quality | tool-eval-bench **~90**/100 (full 69×2, ×4 runs) — parity with the best W4A16 daily |
| **SWE-Bench-Verified** | **69.4%** (347/500, official harness, single attempt) |
| **Terminal-Bench 2.1** | **48.3%** (43/89, terminus-2, default timeouts; **71.7%** on tasks that finished within budget) |
| hardware | 1× RTX 5090 32 GB (+4500 MHz mem OC, 600 W) + Ryzen 9 5900X + 64 GB RAM |
| endpoint | OpenAI-compatible, `http://127.0.0.1:8020/v1` — **loopback-bound by default, no auth**; opt into LAN with `BIND_ADDR=0.0.0.0` behind a firewall/VPN or authenticated proxy |

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

> **Which profile these numbers are from.** Measured on the **no-LMCache** profile (util 0.98, pool 239,436, `mnbt` 4096) — they isolate the *engine*, and they're the numbers to compare against other setups. The same weights and kernels mean the tier profile decodes identically; its narrower `mnbt` (3231) costs a few percent on synthetic deep prefill — quantified in [What removing LMCache changes](docs/LMCACHE.md#what-removing-lmcache-changes).

**Stability — the promotion gauntlet.** Zero-crash across: the ceiling battery at util 0.98 (needle-in-haystack, `pp8192×c8` + `pp30000×c8` killer shapes, 8× distinct ~34K text floods, 8× four-image vision bursts), two *simultaneous* combined waves (16 mixed requests + benchy on a cold engine), a **106-cycle overnight soak** (zero VRAM drift), and 4 full **69×2 tool-evals** under load.

**The operating envelope** (full matrices, protocols, and per-run data: [bench/RESULTS.md](bench/RESULTS.md)):

| | measured |
|---|---|
| prefill, single stream | **13.5K t/s** @8K → 10.1K @30K → 3.5K @180K (shared lane: aggregate flat under concurrency, per-request divides by N) |
| e2e TTFT, cold context | **2.7 s** @30K · 14.1 s @90K · 47 s @180K (was 7.4/27.9/72.4 pre-W4A4) |
| decode, single stream | **~80–160 t/s, content-dependent** (MTP acceptance: creative prose 82, code 158, benchy ~116–140) — flat with depth to 180K, no deep-context crater |
| decode, aggregate | peaks **700–960 t/s** (c8, warm streams); sustained deep-cold c8 is prefill-lane-bound (466 @pp8192, 149 @pp30000) |
| quality | tool-eval-bench **~90**/100 (full 69×2, ×4 runs, pooled 89.8) — parity with the best W4A16 daily (87.8) |

> **Why decode is a range, not a number:** with MTP speculative decoding, the draft head lands more tokens per verify step on predictable output. Same engine, same 600-token budget: *"write a short story"* → 82 t/s, *"create a todo app"* → 158 t/s. If your workload is prose, read the low end; if it's code (this box's job), read the high end.

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

Or skip both builds and pull the **exact validated daily image** (immutable tag — the same bits the agentic benchmarks and the tier soak ran on):

```bash
docker pull ghcr.io/adrienbrault/qwen36-27b-vllm:tiers-lmcfix6-20260722
# digest sha256:51f654b566c54451080164e27a34e5a180c67fa85e3414dabdb340e91b8dccb1
IMAGE=ghcr.io/adrienbrault/qwen36-27b-vllm:tiers-lmcfix6-20260722 ./scripts/serve.sh
```

Then `http://localhost:8020/v1` speaks OpenAI. Every flag is explained inline in [`scripts/serve.sh`](scripts/serve.sh) and in [docs/GOTCHAS.md](docs/GOTCHAS.md).

**Want it without the tiers?** Skip step 2 and run [`./scripts/serve-plain.sh`](scripts/serve-plain.sh) instead — bigger hot pool, no sidecar, base image only. [What you trade](docs/LMCACHE.md#what-removing-lmcache-changes).

## Before you run this — the three landmines

1. **MTP `ns≥2` + fp8 KV IMA-crashes stock vLLM under concurrency** — the image carries a one-line synchronization workaround (below). Single-stream tests hide it; load-test with 3+ parallel streams on day one.
2. **`--no-async-scheduling` is mandatory with MTP** — vLLM's async scheduler corrupts KV under spec decode ([#42655](https://github.com/vllm-project/vllm/issues/42655)).
3. **Verify the KV pool in the launch log**: ~214K on the tier daily, ~239K plain. On the tier daily, 239K means the connector silently didn't attach — the server looks perfectly healthy while every "tier hit" you measure is vLLM's own prefix cache. (The serve scripts now fail closed on this.)

Six more shared ones, plus three tier-specific (the 876 GB disk-fill, the invisible sidecar VRAM, the needle-test discipline) — nine in all: **[docs/GOTCHAS.md](docs/GOTCHAS.md)**.

## Why it needs a patch: MTP × fp8-KV × Blackwell crashes on stock vLLM

> ⚠️ **`ns≥2` MTP with `fp8_e4m3` KV on `sm_120` is a 100%-reproducible illegal-memory-access under concurrency** — a known, still-open upstream bug ([vllm#40756](https://github.com/vllm-project/vllm/issues/40756), same Qwen3.6-27B model; [vllm#35288](https://github.com/vllm-project/vllm/issues/35288)). It crashes at `rejection_sampler.py:267 parse_output → cudaErrorIllegalAddress`. Single-stream and `ns=1` are both clean; `CUDA_LAUNCH_BLOCKING=1` masks it (→ a timing race).

What the image carries is a **locally validated synchronization workaround**, from [PR #42603](https://github.com/vllm-project/vllm/pull/42603): one stream-sync after the MTP draft loop writes its shared cudagraph input buffers, before the draft forward that reads them.

```python
self.input_ids[:batch_size] = input_ids
self.hidden_states[:batch_size] = hidden_states
torch.accelerator.current_stream().synchronize()   # from PR #42603 (closed unmerged)
```

Honesty about its status: **upstream closed that PR unmerged** — maintainers held that these operations should already be stream-ordered and that a forced sync may merely perturb timing rather than fix a proven race, and asked for a true root cause before accepting anything. What *we* can vouch for is empirical: on this exact profile the crash is 100%-reproducible without the sync and has never occurred with it — validated across concurrent c4/c8, deep `pp90000×c4`, and every 69×2 tool-eval, at zero measurable perf cost. The underlying upstream root cause remains unresolved. The multi-day bisection — including every plausible fix that *didn't* work — is in [docs/HISTORY.md](docs/HISTORY.md).

## What's in the patch stack

Three in the base image (below), plus [**six more for the KV tiers**](patches/lmcache/README.md) in the tier image — four on LMCache (fp8 page regrouping ×2, sidecar VRAM, L2 cap enforcement) and two on vLLM (connector-path EAGLE×hybrid hit reduction, Mamba store boundary).

| patch | what it does |
|---|---|
| [`install_pr42603_sync.py`](patches/install_pr42603_sync.py) — from [PR #42603](https://github.com/vllm-project/vllm/pull/42603) (closed unmerged) | **The workaround that makes MTP `ns=4` usable on Blackwell here.** One stream-sync in the MTP draft loop; empirically eliminates the crash above on this profile. Perf-neutral. [Validated numbers](bench/RESULTS.md). |
| FlashInfer 0.6.15 (Dockerfile pip step) | Latest FlashInfer, carrying the `sm_120` GDN/TMA fixes. cu130 AOT cubins aren't published for .15, so it JIT-compiles at runtime — **mount `/root/.cache/flashinfer`** (one build, warm forever). |
| [PR #44993 graft](https://github.com/vllm-project/vllm/pull/44993) | **Structured output that survives thinking.** With a reasoning model, `response_format` json_schema + thinking-on returned EMPTY `content`. Needs `--structured-outputs-config` (see [GOTCHAS.md](docs/GOTCHAS.md)). Two pure-Python files. |

## Alternatives — what this repo is, and isn't

The honest positioning: **the most comprehensively validated single-5090 Qwen3.6-27B setup for concurrent long-context coding agents** — not the maximum-context record (requests cap at 200K where some recipes configure 256K) and not the cherry-picked single-stream decode record. What no alternative we know of documents: a multi-million-token restart-persistent KV hierarchy, deep-context concurrency matrices, and full agentic benchmarks (SWE-Bench-Verified, Terminal-Bench) on the served config.

When something else fits better:

- **[CobraPhil/qwen36-27b-single-5090](https://github.com/CobraPhil/qwen36-27b-single-5090)** — a much simpler Docker-Compose recipe (setup script, checksum-verified model download). W4A16 AutoRound, so no W4A4 prefill lane, and no tiered/persistent KV — but far easier onboarding for a conventional single-user server.
- **[devnen/qwen3.6-windows-server](https://github.com/devnen/qwen3.6-windows-server)** — native Windows, one-click portable releases, no Docker/WSL. The right answer on Windows, and its release engineering (versioned zips, presets, loopback default) is a model we're catching up to.
- **GGUF / llama.cpp routes** — broad frontend compatibility and simple deployment, at a large deficit in prefill and warm-fleet concurrency (measured on this box: [What didn't work](docs/REJECTED.md)).

Our real weaknesses vs. all of them: the nine-patch operational complexity, Linux-only, and numbers measured on a 600 W memory-OC'd card ([DESIGN.md host notes](docs/DESIGN.md#host-notes)). Roadmap items stolen from the reviews of this repo: a `setup.sh` + Compose path with model checksum verification and a machine-readable boot report, and a content-aware MTP `ns=3…6` sweep on coding output.

## How we got here / what didn't work

The docs, by the question you're asking:

- **"Why these weights / where does the VRAM go / what does the host contribute?"** → [docs/DESIGN.md](docs/DESIGN.md) — the W4A4-vs-W4A16 prefill mechanics, the measured 31.35 GiB memory budget, tier cost per token, the OC.
- **"What do the KV tiers buy, and what breaks without the patches?"** → [docs/LMCACHE.md](docs/LMCACHE.md) — the trade table, the 3.4× agentic A/B, the c4 concurrency sweep, and the four rounds of silently-wrong "validated" profiles.
- **"Why doesn't my port of this config work?"** → [docs/GOTCHAS.md](docs/GOTCHAS.md) — every flag with its failure mode, plus the 12 setup gotchas.
- **"How did the config evolve / what's the full bug story?"** → [docs/HISTORY.md](docs/HISTORY.md) — the daily lineage table, the two reversals, the PR #42603 bisection.
- **"Has X been tried?"** → [docs/REJECTED.md](docs/REJECTED.md) — everything rejected, with the number that killed it. Read it before "improving" the config.
- **All result tables and protocols** → [bench/RESULTS.md](bench/RESULTS.md); raw artifacts + pinned rerun commands → [bench/reproduce/](bench/reproduce/README.md); probes in [bench/](bench/README.md).

## License

MIT (see [LICENSE](LICENSE)) for the original work here — docs, benchmarks, scripts, and the patch-installer tooling. Anything derived from vLLM or LMCache — redistributed PR diffs, patch context lines, and the patched files inside built images — **remains Apache-2.0-derived**; the full per-file inventory is in [THIRD_PARTY.md](THIRD_PARTY.md).

## Credits

- [vLLM](https://github.com/vllm-project/vllm), and [PR #42603](https://github.com/vllm-project/vllm/pull/42603)'s author for the draft-loop sync this image carries as a workaround.
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
