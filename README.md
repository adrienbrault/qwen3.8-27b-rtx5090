# Qwen3.8-27B on a single RTX 5090: 381K-token NVFP4 KV pool, XQA decode, restart-proof disk cache, 262K context

A serving config for concurrent long-context coding agents on one 32 GB Blackwell card (`sm_120`): a W4A4 NVFP4 checkpoint, an NVFP4 KV cache with the XQA decode kernel, MTP speculative decoding, vision, and a native disk KV tier on a hard-capped loopback filesystem that survives restarts. Every number was measured on this box on the date given; the raw results directory or FINDINGS round is named next to it.

## The config at a glance (daily since 2026-08-28)

| | |
|---|---|
| model | [gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4) — NVFP4 everywhere incl. the GDN projections and lm_head (ModelOpt), MTP head, vision tower, 18.8 GB; served with the stock Qwen3.8 chat template |
| engine | vLLM **v0.28.0** (release, not nightly) + the [patches-v0280](patches-v0280/) stack: sm120 NVFP4→FA2 routing, the linear-V-scale store overlay, XQA-NVFP4 decode, MTP-drafter full cudagraphs |
| KV cache | `--kv-cache-dtype nvfp4` (E2M1 + FP8 scale per 16 elements), `decode_backend=xqa`; fp8 paths byte-identical to stock |
| hot pool | **381,300 tokens** at util 0.955, max-len 262,144, 8 sequences (fp8 KV on the same engine: ~225K at 200K) |
| disk tier | vLLM's native OffloadingConnector (4 GiB CPU staging → 200 GiB fs tier, `offload_prompt_only`), backed by a **fixed-size loopback ext4 image** — the capacity cap holds by construction. Survives `docker restart`: 40K revisit 1.5–2.6 s vs 5.5–6.2 s cold (R108/R113) |
| spec decode | MTP `ns=4` (the depth curve re-measured on this engine: ns5 ties, ns6 loses — R113), **async scheduling ON** (the v0.28 default; disabling it costs 20–40%) |
| decode (steady state, `scripts/decode_ss.py`, 2026-08-29 on the live daily) | prose c1 124.5; code c1 178 (boot-to-boot 167–206, tracks MTP acceptance 0.43–0.69 — see the variance note in [bench/RESULTS.md](bench/RESULTS.md)); **code c8 1,221 aggregate** (152.6/stream) |
| prefill | ~12.8K t/s at 8K, ~9.3K at 30K, single stream (shared lane) |
| quality | tool-eval-bench 69×4 **90.0 ± 1.4** — identical to the engine with the tier removed and to the previous (LMCache) generation, same-day same-harness (R113/R108) |
| fidelity | prefill-logprob ruler vs the FP8 reference: top-1 0.8895 / ΔNLL 2.25% (parity with fp8 KV); with the V-scale overlay disabled: 0.8552 / 8.82% with zero behavioral symptoms — the falsification run that justifies the fail-closed boot assert (R106) |
| hardware | RTX 5090 32 GB (+4500 MHz memory OC, 600 W), Ryzen 9 5900X, 64 GB RAM, Ubuntu 24.04 |
| endpoint | OpenAI-compatible `http://127.0.0.1:8020/v1`, served as `qwen3.8-27b` (alias `qwen3.6-27b`); loopback, no auth |

Source: FINDINGS R104–R113, results dirs `2026-08-28-r10*`. The previous generation (vLLM 0.26 nightly + LMCache DRAM/NVMe tiers, daily 2026-08-21→28) is fully documented in [docs/HISTORY.md](docs/HISTORY.md) and remains reproducible from [patches/rc4](patches/rc4/) + [patches-nvfp4kv](patches-nvfp4kv/).

## What it optimizes for

A few coding agents with 8K–100K+ contexts, interactive chat, the occasional image, on one always-on box. Ranked:

1. Correctness of the cache. An engine that answers fluently from corrupted KV is worse than a slower one. Nothing becomes the daily without the gauntlet: the fidelity ruler vs the FP8 reference (behavioral probes are provably blind to KV-layout bugs — twice measured here), depth needles cold and warm, a needle retrieved through disk-tier eviction and after a container restart, and a full tool-eval at ×4 trials.
2. Context capacity that passes rule 1.
3. Latency in the agent regime: prefill on the FP4 tensor cores, single-stream decode through MTP + XQA.
4. Everything on at once: vision, 262K context, speculative decoding, reasoning, structured outputs, tool calling.

Non-goals: maximum aggregate throughput for many shallow users, multi-GPU, minimum VRAM.

## Benchmarks

**Checkpoint provenance:** the parent `-RTX5090` HF repo later swapped its `lm_head` to BF16 and republished the original NVFP4-head recipe as [`-LMHead4`](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4) — verified bit-identical to what this stack serves. Download `-LMHead4` to reproduce.

Current-generation tables (v0.28 audition, XQA, the disk tier, tuning ladder, decode-variance analysis): [bench/RESULTS.md](bench/RESULTS.md), sections dated 2026-08-28/29. Agentic results (latest runs): SWE-Bench Verified **331/500 = 66.2%** (R2E-Gym scaffold, function calling on — note its gate keys on the served model name), Terminal-Bench 2.1 **50/89 = 56.2%**, both with full context in [bench/RESULTS.md](bench/RESULTS.md); TB's binding constraint is wall-clock, so decode throughput dominates quant fidelity there. Checkpoint-selection history (four NVFP4 checkpoints on one ruler) and the previous generation's tables: same file, earlier sections, and [docs/HISTORY.md](docs/HISTORY.md).

## Quick start

```bash
# 0. one-time: the hard-capped backing store for the disk tier
sudo bash scripts/setup-native-l2.sh

# 1. the image: vLLM v0.28.0 + the sm120 NVFP4 stack (overlay AOT-built; no GPU needed to build)
cd patches-v0280 && docker build -f Dockerfile.v0280-nvfp4kv -t vllm-qwen38:v0280-nvfp4kv . && cd ..

# 2. serve
MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4-RTX5090-LMHead4 PORT=8020 NAME=vllm-27b ./scripts/serve-v0280-daily.sh
```

The serve script fails closed on every load-bearing property: the overlay-ACTIVE line, `decode_backend=xqa`, connector init, the pool band, a MemAvailable gate before the engine swap, and an orphaned-shm sweep. Every flag is annotated inline.

## Things that bite

1. **The V-scale store overlay is required on sm120, and you cannot see it missing.** vLLM's in-tree writer swizzles V block scales for SM100; the FA2/XQA readers address them linearly. With the stock writer the engine is fluent, passes needles, and runs at ΔNLL 8.82% vs 2.25% (top-1 0.855 vs 0.890) — measured directly by booting with the overlay disabled (R106). The launcher refuses to serve without the ACTIVE line; keep it that way.
2. **Never pass `--no-async-scheduling` on vLLM ≥ 0.28.** The old advice (it guarded a 0.26-era async×spec bug) now *disables a default optimization*: +21% prose c1, +29% code c1 measured from dropping the flag (R104e).
3. **`prompt_logprobs` probes are blind to decode kernels** — they never execute a decode step. Validating a decode-path change (like XQA) needs a decode-path instrument (`scripts/decode_fidelity.py`) plus a task-accuracy A/B; prefill rulers will read bit-identical no matter what the decode kernel does (R107b).
4. **The OffloadingConnector leaks its 4 GiB staging mmap** (`/dev/shm/vllm_offload_*.mmap`) past `docker rm -f`. Four engine swaps quietly ate 16 GiB of host RAM. The serve script sweeps orphans (fuser-guarded) before every boot.
5. **Engine swaps can transiently exceed host RAM**: `docker rm -f` returns before the old engine's memory is reaped, and booting the next engine into that window global-OOMed the host (the OOM killer only reaps small high-`oom_score_adj` victims, so the box thrashed for hours). The serve script gates on MemAvailable before starting the container.
6. **Offload decode-phase KV and you pay for it in agentic quality**: the default (offload everything) cost 1.8 tool-eval points in write stalls; `offload_prompt_only` restored parity, since prefix reuse only ever hits prompt blocks (R113).
7. **Don't trust the tier's own capacity limit** — back it with a fixed-size filesystem so the cap holds by construction. (An earlier generation's cache once grew to 876 GB past a 60 GB config cap.)
8. **Single-stream decode swings ±10% boot-to-boot** with speculative acceptance (0.43–0.69 on identical prompts). A/B decode within one boot, or normalize by acceptance.

Previous-generation gotchas (LMCache chunk=block, sidecar shm, util ceilings): [docs/GOTCHAS.md](docs/GOTCHAS.md).

## Patch stacks

| stack | what | status |
|---|---|---|
| [patches-v0280/](patches-v0280/README-sm120-nvfp4.md) | **Current.** vLLM v0.28.0: PR #49891 re-rebase (0101), linear-V-scale overlay (0102), XQA-NVFP4 decode (0103), drafter full cudagraphs (0104), DFlash2 quantized-draft loader fix, DFlash2 non-causal A/B diff. Per-hunk rationale in the README; provenance in [THIRD_PARTY.md](THIRD_PARTY.md). | daily |
| [patches/rc4/](patches/rc4/README.md) + [patches-nvfp4kv/](patches-nvfp4kv/README.md) | The 0.26/0.27 generation: LMCache tier fixes and the original sm120 NVFP4 overlay. | superseded, reproducible |

## Rejected

[docs/REJECTED.md](docs/REJECTED.md) lists every rejected configuration with the number that killed it. Read it before changing the config.

## Docs

- [docs/GOTCHAS.md](docs/GOTCHAS.md) — every flag with its failure mode.
- [docs/DESIGN.md](docs/DESIGN.md) — why W4A4 weights, where the VRAM goes.
- [docs/NVFP4KV.md](docs/NVFP4KV.md) — the NVFP4 KV cache and the overlay.
- [docs/HISTORY.md](docs/HISTORY.md) — daily lineage since 2026-06, reversals included; the LMCache-tier generation lives here.
- [bench/RESULTS.md](bench/RESULTS.md) — all tables, newest first; [bench/](bench/README.md) — probes and commands.

## License

MIT ([LICENSE](LICENSE)) for the original work here. Anything derived from vLLM, LMCache or FlashInfer — redistributed PR diffs, patch context, patched files inside built images — stays Apache-2.0-derived; per-file inventory in [THIRD_PARTY.md](THIRD_PARTY.md).

## Credits

- [vLLM](https://github.com/vllm-project/vllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [LMCache](https://github.com/LMCache/LMCache) (previous generation).
- ch2lab for [vLLM PR #49891](https://github.com/vllm-project/vllm/pull/49891) (sm120 nvfp4 KV routing).
- drowzeys for the linear-V-scale writer fix ([DGX Spark repo](https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-27B-ADay777Ablit-NVFP4-A4Q-NVFP4-KV-4M-KV-token-pool-MTP3-Single-DGX-Spark)).
- The author of [vllm#49011](https://github.com/vllm-project/vllm/issues/49011) for demonstrating XQA-NVFP4 decode + FA2 prefill on sm120, and [hikarioyama](https://github.com/hikarioyama/vllm-nvfp4-kv-sm120) for the FA2 SF-stride prior art.
- [seanyourhighness](https://github.com/seanyourhighness/vllm-sm12x-nvfp4-dflash2) for the DFlash2-with-NVFP4 overlay this repo's non-causal route and acceptance work port from, and for validating `--kv-cache-memory-bytes` pinning.
- [sakamakismile](https://huggingface.co/sakamakismile/Qwen3.8-27B-MTP-NVFP4) for the W4A4 export lineage; z-lab and inco.ai for DFlash2 and [vLLM PR #52816](https://github.com/vllm-project/vllm/pull/52816).

Full per-file provenance: [THIRD_PARTY.md](THIRD_PARTY.md).
