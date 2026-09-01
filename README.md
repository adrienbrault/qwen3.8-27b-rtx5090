# Qwen3.8-27B on RTX 5090s: 262K context on one card — 654K-token KV pool and ~320 t/s decode on two

Serving configs for concurrent long-context coding agents on Blackwell consumer cards (`sm_120`): a W4A4 NVFP4 checkpoint, quantized KV with the XQA decode kernel, speculative decoding (MTP on one card, DFlash2 across two), vision, and a native disk KV tier on a dedicated Gen5 NVMe partition that survives restarts. Every number was measured on this box on the date given; the raw results directory or FINDINGS round is named next to it.

## The checkpoint changed (2026-09-02): RedHatAI NVFP4 replaces gittensor

The daily now serves [`RedHatAI/Qwen3.8-27B-NVFP4`](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4) (llm-compressor, 303 modules kept at 8-bit, FP8 `lm_head`) on the unchanged TP=2 DFlash2 shape — [scripts/serve-r156-daily.sh](scripts/serve-r156-daily.sh). It was chosen by a bf16-anchored fidelity ladder, not by a task benchmark, because every task gate we own is blind at this effect size (GSM8K at n=250 cannot resolve below ~8 pp). Method, all nine checkpoints, and the caveats: [bench/RESULTS.md](bench/RESULTS.md#the-checkpoint-decision-a-bf16-anchored-fidelity-ladder-2026-09-01) and the decision sheet in [docs/R156-DECISION.md](docs/R156-DECISION.md).

| | gittensor (previous daily) | **RedHatAI (daily since 2026-09-02)** |
|---|---|---|
| perplexity gap vs bf16, 725K teacher-forced positions of raw text | +4.46% (last of 9 configs) | **+0.38%** |
| bf16-generated agentic turns (tool/code/reason/prose, 58K positions): top-1 agreement / flips where bf16 was moderately sure | 92.6% / 8.6% | **95.9% / 3.4%** |
| the same, with 4-bit KV instead of fp8 | — | 95.6% / 3.8% |
| spec-ON decode c1 (llama-benchy, T=0.6, pp2048/tg256) | 338.7 t/s | 318.8 (−6%) |
| spec-ON decode c8 code, steady state | 1,299 | 1,212 (−7%) |
| forward pass alone (spec off), c1 | 129.6 | 105.5 (−18.6%) |
| prefill @2K | 10,190 t/s | 8,741 (−14%) |
| KV pool @262K max-len | 746,849 | 654,491 (−12%) |
| tool-eval ×4 / GSM8K T=0 ×120 / needles | 90.0 ± 1.4 / 0.8583 / 9/9 | 90.2 ± 1.0 / 0.8583 / 9/9 |

Three things the ladder taught that generalize: the **quantizer recipe matters more than bit-width** (unsloth, kelnei and RedHat reach +0.37% at the same 4-bit width where gittensor sits at +4.46%); a **4-bit `lm_head` costs ~0.85 pp** on its own (controlled pair); **fp8 KV is nearly free (+0.13 pp)** and does not compound with context out to 171K, while 4-bit KV costs +0.76 pp. And one that bit: **draft acceptance measures drafter–target agreement, not quality** — the quantized syv-ai drafter beats the bf16 original on both checkpoints here, but a mid-fidelity QAT checkpoint lost 26% code decode to drafter mismatch. Re-run the drafter A/B for every checkpoint switch.

Two serving notes from the switch: the native disk tier's namespace hash is built from the model *path*, dtype and parallelism — not the weights — so a checkpoint swap would silently serve the previous checkpoint's KV blocks; the launcher now stamps the tier with the checkpoint name and wipes on change. And the boot-time pool is bimodal on this shape (654,491 or 628,798; the profiler's peak-activation estimate alternates between 1.38 and 1.98 GiB), so the pool band is 620–690K.

## Three serving configs (measured 2026-08-31 on the previous gittensor checkpoint, same day, same harness)

The [gittensor NVFP4 checkpoint](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4) served in three shapes on this box until 2026-09-02; only the DFlash2 column has been re-measured on RedHat (table above). **TP=1** is the recipe this repo grew up on — everything fits one 32 GB card. **TP=2 DFlash2** is the served daily since the box gained a second 5090 (PCIe Gen5 x8/x8, P2P-enabled driver, custom allreduce) — the latency shape. **TP=2 MTP** is the TP=1 recipe stretched across both GPUs, nothing changed but `TP=2` — the capacity/concurrency shape. All run vLLM v0.28.0 + [patches-v0280](patches-v0280/), the native disk KV tier, and memory-OC'd cards (+15% measured DRAM bandwidth).

| | **TP=1** (one 5090) | **TP=2 DFlash2** (the daily) | **TP=2 MTP** |
|---|---|---|---|
| recipe | nvfp4 KV + XQA decode + MTP ns4, util 0.955 — [scripts/serve-v0280-daily.sh](scripts/serve-v0280-daily.sh) | fp8 KV + DFlash2 ns9 ([syv-ai W4A16 drafter](https://huggingface.co/syv-ai)), util 0.92, `NCCL_P2P_LEVEL=SYS` — [scripts/serve-r134-daily.sh](scripts/serve-r134-daily.sh) (gittensor; the RedHat daily is [serve-r156-daily.sh](scripts/serve-r156-daily.sh)) | nvfp4 KV + XQA decode + MTP ns4, util 0.90 — [serve-v0280-daily.sh](scripts/serve-v0280-daily.sh) with `TP=2` |
| KV pool @ 262K max-len | 381,300 tokens | 746,849 | **1,508,519 (~4x one card)** |
| decode, prose c1 | 130.1 t/s | **178.1** | 157.5 |
| decode, code c1 | 175.0 | **298.9** (runs span 293–385) | 225.3 |
| decode, prose c8 aggregate | 941 | 961 | **1,116** |
| decode, code c8 aggregate | 1,187 | 1,289 | **1,349** |
| decode, code c16 aggregate | — (spec-decode admission cap) | 1,522 | **2,007** |
| decode @30K context | 131.9 | **168.7** | 148.3 |
| decode @100K | 106.7 | **174.4** | 137.5 |
| decode @200K | 94.3 | **135.6** | 135.1 |
| prefill @8K | **11.9K t/s** | 9.3K | 9.0K |
| prefill @30K | 9.3K | **9.8K** | 9.1K |
| prefill @100K | 4.7K | **7.0K** | 6.3K |
| prefill @200K | 2.3K | **3.9K** | 3.5K |
| tool-eval ×4 | 89.2 ± 1.7 | 89.8 ± 1.3 | 90.2 ± 1.0 |
| GSM8K T=0 ×120 | 0.8417 | 0.8583 | 0.8333 |
| retrieval needles | 4/4 | 4/4 | 4/4 |

Reading the columns: **DFlash2-TP2 is the latency shape** — fastest single-stream decode from the surface to 100K deep (+37–71% over one card), and it wins prefill everywhere past 8K. **MTP-TP2 is the capacity shape** — a 1.5M-token KV pool and the best aggregate throughput at deep concurrency (2,007 t/s at c16, where DFlash2's low acceptance costs it) — and at batched prose (1,116 vs ~950 at c8, same mechanism). Quality is config-invariant. Two footnotes the numbers force: DFlash2's speed is a random variable (acceptance runs 0.12–0.30 with content — the code-c1 band is real, and its @200K decode converges to MTP's as acceptance decays with depth), and the @200K prefills are cold-cache — a warm disk-tier revisit cut MTP-TP2's 200K TTFT from 56.5 s to 39.5 s on the same prompt.

Host: ASRock X870 Taichi Creator, Ryzen 7 9800X3D, 64 GB DDR5-6000, model weights and the KV disk tier both on a Gen5 x4 NVMe (fio post-format: 10.4 GB/s read, 9.8 GB/s write, 1.38M IOPS), Ubuntu 24.04 HWE; GPUs: ASUS 600 W + HP OEM 575 W at Gen5 x8/x8 with the P2P driver ([THIRD_PARTY](THIRD_PARTY.md)). Why the split wins land where they do: speculative decoding with *low* acceptance is weight-bandwidth-bound — exactly what a second card doubles — while high-acceptance MTP amortizes weight reads and converts the second card into KV space and admission headroom instead (bench/RESULTS.md, R130–R143). Endpoint: OpenAI-compatible `:8020/v1`, served as `qwen3.8-27b`.

Raw results: `results/2026-08-31-r142-matrix` on the host; history in [bench/RESULTS.md](bench/RESULTS.md).

## What it optimizes for

A few coding agents with 8K–100K+ contexts, interactive chat, the occasional image, on one always-on box. Ranked:

1. Correctness of the cache. An engine that answers fluently from corrupted KV is worse than a slower one. Nothing becomes the daily without the gauntlet: the fidelity ruler vs the FP8 reference (behavioral probes are provably blind to KV-layout bugs — twice measured here), depth needles cold and warm, a needle retrieved through disk-tier eviction and after a container restart, and a full tool-eval at ×4 trials.
2. Context capacity that passes rule 1.
3. Latency in the agent regime: prefill on the FP4 tensor cores, single-stream decode through speculative decoding (DFlash2 ns9 on the TP=2 daily; MTP + XQA on one card).
4. Everything on at once: vision, 262K context, speculative decoding, reasoning, structured outputs, tool calling.

Non-goals: maximum aggregate throughput for many shallow users, multi-GPU, minimum VRAM.

## Benchmarks

**Checkpoint provenance:** the daily is [`RedHatAI/Qwen3.8-27B-NVFP4`](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4) since 2026-09-02, served with its own chat template. The 2026-08-22→09-02 daily was gittensor's NVFP4 export; that parent `-RTX5090` HF repo later swapped its `lm_head` to BF16 and republished the original NVFP4-head recipe as [`-LMHead4`](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4) — verified bit-identical to what this stack served then. Every 2026-08 number below was measured on that checkpoint.

Current-generation tables (v0.28 audition, XQA, the disk tier, tuning ladder, decode-variance analysis): [bench/RESULTS.md](bench/RESULTS.md), sections dated 2026-08-28/29. Agentic results (latest runs): SWE-Bench Verified **331/500 = 66.2%** (R2E-Gym scaffold, function calling on — note its gate keys on the served model name), Terminal-Bench 2.1 **50/89 = 56.2%**, both with full context in [bench/RESULTS.md](bench/RESULTS.md); TB's binding constraint is wall-clock, so decode throughput dominates quant fidelity there. Checkpoint-selection history (four NVFP4 checkpoints on one ruler) and the previous generation's tables: same file, earlier sections, and [docs/HISTORY.md](docs/HISTORY.md).

## Quick start

```bash
# 0. one-time: the hard-capped backing store for the disk tier
sudo bash scripts/setup-native-l2.sh

# 1. the image: vLLM v0.28.0 + the sm120 NVFP4 stack (overlay AOT-built; no GPU needed to build)
cd patches-v0280 && docker build -f Dockerfile.v0280-nvfp4kv -t vllm-qwen38:v0280-nvfp4kv . && cd ..

# 2. serve
# one card (nvfp4 KV + MTP):
MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4 PORT=8020 NAME=vllm-27b ./scripts/serve-v0280-daily.sh
# two cards, the daily (fp8 KV + DFlash2; edit MODEL/DRAFT paths at the top):
./scripts/serve-r156-daily.sh
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
- [RedHatAI](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4) for the daily's weights (llm-compressor recipe), and [unsloth](https://huggingface.co/unsloth) / [kelnei](https://huggingface.co/kelnei) for the two checkpoints that tie it on fidelity; [sakamakismile](https://huggingface.co/sakamakismile/Qwen3.8-27B-MTP-NVFP4) for the W4A4 export lineage; z-lab and inco.ai for DFlash2 and [vLLM PR #52816](https://github.com/vllm-project/vllm/pull/52816).

Full per-file provenance: [THIRD_PARTY.md](THIRD_PARTY.md).
