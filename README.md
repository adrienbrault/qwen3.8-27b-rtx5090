# Qwen3.8-27B on RTX 5090s: 262K context on one card — 719K-token KV pool and ~300 t/s code decode on two

Serving configs for concurrent long-context coding agents on Blackwell consumer cards (`sm_120`): a W4A4 NVFP4 checkpoint, quantized KV with the XQA decode kernel, speculative decoding (MTP on one card, DFlash2 across two), vision, and a native disk KV tier on a dedicated Gen5 NVMe partition that survives restarts. Every number was measured on this box on the date given; the raw results directory or FINDINGS round is named next to it.

## Three serving configs, one checkpoint (all numbers re-measured 2026-08-31, same day, same harness)

The same [gittensor NVFP4 checkpoint](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4) serves in three shapes on this box. **TP=1** is the recipe this repo grew up on — everything fits one 32 GB card. **TP=2 DFlash2** is the served daily since the box gained a second 5090 (PCIe Gen5 x8/x8, P2P-enabled driver, custom allreduce) — the latency shape. **TP=2 MTP** is the TP=1 recipe stretched across both GPUs, nothing changed but `TP=2` — the capacity/concurrency shape. All run vLLM v0.28.0 + [patches-v0280](patches-v0280/), the native disk KV tier, and memory-OC'd cards (+15% measured DRAM bandwidth).

| | **TP=1** (one 5090) | **TP=2 DFlash2** (the daily) | **TP=2 MTP** |
|---|---|---|---|
| recipe | nvfp4 KV + XQA decode + MTP ns4, util 0.955 — `scripts/serve-v0280-daily.sh` | fp8 KV + DFlash2 ns9 ([syv-ai W4A16 drafter](https://huggingface.co/syv-ai)), util 0.90, `NCCL_P2P_LEVEL=SYS` — `scripts/serve-r134-daily.sh` | nvfp4 KV + XQA decode + MTP ns4 at `TP=2`, util 0.90 |
| KV pool @ 262K max-len | 381,300 tokens | 719,420 | **1,508,519 (~4x one card)** |
| decode, prose c1 | 130.1 t/s | **178.1** | 157.5 |
| decode, code c1 | 175.0 | **298.9** (runs span 293–385) | 225.3 |
| decode, code c8 aggregate | 1,187 | 1,289 | **1,349** |
| decode, code c16 aggregate | — (spec-decode admission cap) | 1,522 | **2,007** |
| decode @30K context | 131.9 | **168.7** | 148.3 |
| decode @100K | 106.7 | **174.4** | 137.5 |
| decode @200K | 108.9 | 135.6 | 135.1 |
| prefill @8K | **11.9K t/s** | 9.3K | 9.0K |
| prefill @30K | 9.3K | **9.8K** | 9.1K |
| prefill @100K | 4.7K | **7.0K** | 6.3K |
| prefill @200K | 3.2K | **3.9K** | 3.5K |
| tool-eval ×4 | 89.2 ± 1.7 | 89.8 ± 1.3 | 90.2 ± 1.0 |
| GSM8K T=0 ×120 | 0.8417 | 0.8583 | 0.8333 |
| retrieval needles | 4/4 | 4/4 | 4/4 |

Reading the columns: **DFlash2-TP2 is the latency shape** — fastest single-stream decode from the surface to 100K deep (+37–71% over one card), and it wins prefill everywhere past 8K. **MTP-TP2 is the capacity shape** — a 1.5M-token KV pool and the best aggregate throughput at deep concurrency (2,007 t/s at c16, where DFlash2's low acceptance costs it). Quality is config-invariant. Two footnotes the numbers force: DFlash2's speed is a random variable (acceptance runs 0.12–0.30 with content — the code-c1 band is real, and its @200K decode converges to MTP's as acceptance decays with depth), and the @200K prefills are cold-cache — a warm disk-tier revisit cut MTP-TP2's 200K TTFT from 56.5 s to 39.5 s on the same prompt.

Host: ASRock X870 Taichi Creator, Ryzen 7 9800X3D, 64 GB DDR5-6000, model weights and the KV disk tier both on a Gen5 x4 NVMe (fio post-format: 10.4 GB/s read, 9.8 GB/s write, 1.38M IOPS), Ubuntu 24.04 HWE; GPUs: ASUS 600 W + HP OEM 575 W at Gen5 x8/x8 with the P2P driver ([THIRD_PARTY](THIRD_PARTY.md)). Why the split wins land where they do: speculative decoding with *low* acceptance is weight-bandwidth-bound — exactly what a second card doubles — while high-acceptance MTP amortizes weight reads and converts the second card into KV space and admission headroom instead (bench/RESULTS.md, R130–R143). Endpoint: OpenAI-compatible `:8020/v1`, served as `qwen3.8-27b`.

Raw results: `results/2026-08-31-r142-matrix` on the host; history in [bench/RESULTS.md](bench/RESULTS.md).

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
8. **Single-stream decode swings ±10% boot-to-boot** with speculative acceptance (0.43–0.69 on identical prompts) — the mechanism is T>0 sampling-content divergence through content-dependent acceptance (T=0 outputs are bit-identical across boots), decorrelated by batching timing. A/B decode within one boot, normalize by accepted-tokens-per-step, or probe at T=0.
9. **DFlash2 drafts read the target's KV non-causally** — a third read path beyond XQA-decode and FA2-causal-prefill, and on this FlashInfer build it faults on NVFP4 pages (five-theory falsification ledger in [bench/RESULTS.md](bench/RESULTS.md)). DFlash2 therefore pairs with fp8 KV (where it holds the code records); the NVFP4 daily uses MTP, which is purely causal.

Previous-generation gotchas (LMCache chunk=block, sidecar shm, util ceilings): [docs/GOTCHAS.md](docs/GOTCHAS.md).

## Patch stacks

| stack | what | status |
|---|---|---|
| [patches-v0280/](patches-v0280/README-sm120-nvfp4.md) | **Current.** vLLM v0.28.0, 0101–0113: NVFP4 routing + overlay + XQA decode + drafter graphs (0101–0104), non-causal/dflash work (0105/0107/0109/0113), sampling guard (0106), GDN hardening (0108), ReplaySSM chunked verify (0111, OFF-default), XQA verify (0112, OFF-default). Per-hunk READMEs in the directory; provenance in [THIRD_PARTY.md](THIRD_PARTY.md). | daily |
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
