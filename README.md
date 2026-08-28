# Qwen3.8-27B on a single RTX 5090: 312K-token KV pool, restart-proof cache tiers, 262K context

This is a serving config for concurrent long-context coding agents on one 32 GB Blackwell card (`sm_120`): a W4A4 NVFP4 checkpoint, an NVFP4 KV cache, MTP speculative decoding, vision, and a DRAM/NVMe KV tier that survives restarts. Every number here was measured on this box on the date given; the raw results directory or FINDINGS round is named next to it.

## The config at a glance (daily since 2026-08-21)

| | |
|---|---|
| model | [gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4) — NVFP4 everywhere incl. the GDN projections (ModelOpt), MTP head, vision tower, 18.8 GB; served with the stock Qwen3.8 chat template (its bundled one breaks `qwen3_xml` tool parsing). Daily since 2026-08-22; previous: [sakamakismile/Qwen3.8-27B-MTP-NVFP4](https://huggingface.co/sakamakismile/Qwen3.8-27B-MTP-NVFP4) (GDN in bf16, 20.6 GB) |
| engine | vLLM nightly `ba07e4a48` (2026-08-21) on the V2 model runner, FlashInfer 0.6.17, two local patch stacks ([rc4](patches/rc4/) for the tiers, [nvfp4kv](patches-nvfp4kv/) for the KV cache) |
| KV cache | `--kv-cache-dtype nvfp4` (E2M1 values + one FP8 scale per 16 elements, 0.5625 B/elt), unified hybrid block 2864 tokens |
| hot pool | **312,189 tokens** at util 0.93, max-len 262,144 (the model's limit; 1.19× at full length), 8 sequences (fp8 KV on the same engine at 200K: 208,450). Depth needles cold + warm at 100K / 180K / 261.7K prompt tokens: all hit (2026-08-21, results/2026-08-21-r82-ladder). |
| tiers | LMCache 0.5.4rc4: 24 GiB pinned DRAM + 200 GiB NVMe (`fs_native`), chunk 2864; the NVMe tier survives restarts. Measured density on nvfp4 pages: ~73 KB/token incl. the GDN state, so 200 GB holds ~2.7M tokens (about 9× the hot pool); LRU eviction at the 80% watermark is silent and keeps the tier bounded (2026-08-21 soak, results/2026-08-21-soak-nvfp4-l2) |
| spec decode | MTP `ns=4`, `--no-async-scheduling`, `--mamba-cache-mode align` |
| decode (steady state, prose / code) | c1 124 / 183 t/s; c4 511 / 639 aggregate; c8 891 aggregate (prose); 103 t/s single-stream at 100K context |
| prefill | 12.8K t/s at 8K, 9.3K at 30K, single stream (the prefill lane is shared; per-request divides by N) |
| quality | tool-eval-bench 69×2 **92 ± 1.4**; fp8-tier control 69×4 90.8 ± 0.5 |
| retrieval | needles 9K→100K cold + warm 10/10 + 10/10; restart-proof L2 revisit 40K 6.3 → 2.5 s, 60K 11.6 → 3.3 s; 15/15 + 15/15 under 4 concurrent loaders |
| hardware | RTX 5090 32 GB (+4500 MHz memory OC, 600 W), Ryzen 9 5900X, 64 GB RAM, Ubuntu 24.04 |
| endpoint | OpenAI-compatible `http://127.0.0.1:8020/v1`, served as `qwen3.8-27b` (alias `qwen3.6-27b` kept for old clients); loopback by default, no auth |

Source: FINDINGS R79 and R81, results dirs `2026-08-21-qwen38-tier-v2` and `2026-08-21-qwen38-tiers-nvfp4kv`.

## What it optimizes for

A few coding agents with 8K–100K+ contexts, interactive chat, the occasional image, on one always-on box. Ranked:

1. Correctness of the cache. An engine that answers fluently from corrupted KV is worse than a slower one. Nothing becomes the daily without the gauntlet in [Benchmarks](#benchmarks): depth needles cold and warm, a needle retrieved after a container restart, the same under concurrent load, a cold 8×24K burst, a vision burst, and a full tool-eval.
2. Context capacity that passes rule 1. The NVFP4 KV cache is the first 4-bit KV format that did on this hybrid; the tiers add DRAM and NVMe below it.
3. Latency in the agent regime: prefill (W4A4 on the FP4 tensor cores) and single-stream decode (MTP).
4. Everything on at once: vision, 262K context, speculative decoding, reasoning, structured outputs, tool calling.

Non-goals: maximum aggregate throughput for many shallow users, multi-GPU, minimum VRAM.

## How the 2026-08-21 daily came about

The session started from a 5-day outage (self-inflicted: procps `kill -TERM -<pgid>` parses to `kill(-1)`), found that a 20% SWE-Bench regression was the harness silently disabling function calling, tried DFlash2 instead of MTP (rejected: MTP on the same runner matched it at 2.5× the context — [docs/DFLASH2.md](docs/DFLASH2.md)), and in doing so found that the V2 model runner plus the newer nightly gave MTP +16–20% decode ([docs/V2RUNNER.md](docs/V2RUNNER.md)). The NVFP4 KV cache then went from "needs an LMCache page-regrouping project" to "works with one config line" once the unified block (2864 tokens) was read off the engine log ([docs/NVFP4KV.md](docs/NVFP4KV.md)).

## Benchmarks

**Checkpoint provenance (2026-08-28):** the parent `-RTX5090` repo later swapped its `lm_head` to BF16 and republished the original NVFP4-head recipe as [`-LMHead4`](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4) — verified bit-identical (per-tensor md5 + shard sizes) to the 2026-08-22 download this stack serves. To reproduce these results, download `-LMHead4`, not the parent. Historical mentions of `-RTX5090` below refer to this same recipe.

**Checkpoint switch (2026-08-22):** the daily moved from saka to [gittensor `-RTX5090`](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4), which also quantizes the GDN projections: KV pool 388K vs ~310K at 262K, +20–25% decode at every concurrency, prefill parity; cost: tool-eval 69×4 89.8 ± 1.3 vs saka's 92 ± 1.4, SWE-Bench parity on a same-tasks 50-task slice (22/30 vs 37/50 pace). [Mantrah `-GDN`](https://huggingface.co/Mantrah/Qwen3.8-27B-NVFP4-GDN) measured the same quality (89.5, 25/34) with a smaller pool (348K). Table in [bench/RESULTS.md](bench/RESULTS.md#checkpoint-ab-on-the-daily-engine-2026-082122-results2026-08-21-radixark-ab-results2026-08-22-sweep-ab). On 2026-08-23 all four NVFP4 checkpoints (gittensor, Mantrah, saka, unsloth) were measured on one ruler — steady-state decode, pool, token fidelity vs the FP8 reference (`scripts/fidelity.py`), GSM8K and IFEval at n≥541: gittensor keeps the daily; saka's quality edge is a calibration shift toward agentic text, not lower quantization loss. Table in [bench/RESULTS.md](bench/RESULTS.md#the-four-nvfp4-checkpoints-on-one-ruler-2026-08-23-results2026-08-23-nvfp4-quant-sweep-results2026-08-23-fidelity).

Tool: [llama-benchy](https://github.com/eugr/llama-benchy) 0.3.8 unless stated. Memory-overclocked card (+4500 MHz VRAM, about 15% more bandwidth than stock); decode is bandwidth-bound, so expect up to ~15% lower decode at stock clocks. Full tables: [bench/RESULTS.md](bench/RESULTS.md).

**Decode, steady state (2026-08-23, gittensor daily, `scripts/decode_ss.py`: throughput from vLLM's own counters over the window where all streams are decoding, `min_tokens` = 1024, median of 3 runs, run spread 3–6%):**

| | c1 | c2 | c4 | c8 | c1 at 30K context | c1 at 100K context |
|---|---|---|---|---|---|---|
| prose (long essay), aggregate t/s | 124 | 270 | **511** | **891** | 115 | 103 |
| prose, per stream | 124 | 135 | 128 | 111 | 115 | 103 |
| prose, MTP accepted per draft token | 0.32 | 0.38 | 0.38 | 0.37 | 0.31 | 0.32 |
| code (implement a small library), aggregate | 183 | — | **639** | — | — | — |
| `vllm bench serve`, random 1024→512 tokens, closed loop | — | — | 602 (median TPOT 5.1 ms, TTFT 221 ms) | — | — | — |

Two things this table makes explicit that the earlier ones hid. **Content sets single-stream decode** through MTP acceptance: the same engine does 124 t/s on prose (0.32 accepted per draft token) and 183 on code (0.61); llama-benchy's synthetic prompts land at ~170. **Concurrency scales near-linearly to eight streams** — 891 t/s aggregate at 111 per stream. Depth costs about 17% at 100K on prose.

**Why the earlier concurrent numbers were wrong.** Until 2026-08-23 the tables here quoted llama-benchy's "t/s (total)" at c4/c8 (339–451 and 353–487 for this class of stack). That figure divides generated tokens by the whole window, and with `tg512` the window is mostly the staggered prefill ramp-up and the ramp-down, during which fewer than c streams are decoding — the same stack read anywhere from 297 to 463 at c4 from one run to the next. benchy's *peak* column (instantaneous rate with every stream live) was always close to the steady state. Prefill and TTFT from benchy are single-shot timings and stand. `vllm bench serve` agrees with the probe (602 vs 511–639 depending on content) when pointed at the raw `/v1/completions` endpoint; against the chat endpoint with a reasoning parser it counts only `content` tokens, so the thinking phase shows up as 25 s of "TTFT" and a 70 t/s throughput — do not use it that way.

**Prefill, single stream (R81, mnbt 5727):** 12.8K t/s at 8K, 9.3K at 30K. The fp8 tier profile (mnbt 3231) reads 9.7K / 8.9K on the same engine; the plain profile at mnbt 4096 reads 13.0K / 9.7K.

**Retrieval and cache correctness (R81):** depth needles 9K / 20K / 40K / 60K / 100K, two samples each, exact match of a random secret, cold and warm: 10/10 + 10/10; warm revisits 0.5 s at 9K to 1.2–2.0 s at 100K (cold 27 s). Restart-proof: store 40K and 60K, `docker restart` the engine, revisit: 6.3 → 2.5 s and 11.6 → 3.3 s, both correct. Under 4 concurrent 20K loaders at 32K / 48K / 64K × 5: 15/15 cold, 15/15 warm. Cold burst of 8 × 24K: 8/8.

**Quality:** tool-eval-bench 69×2 at T=0.6, reasoning effort medium: 92 ± 1.4 on the daily; 69×4 on the fp8 tier profile the same day: 90.8 ± 0.5; the previous image on the V2 runner: 91.2 ± 2.1. The 2026-07 era's 89.0 ± 1.4 and the 2026-08-15 92.5 ± 0.7 (n=2) sit in the same band. Lost points land in the same categories every run (Instruction Following, Context & State, Safety & Boundaries).

**Agentic benchmarks.** SWE-Bench Verified on the 2026-08-21 daily (saka checkpoint, NVFP4 KV + tiers + V2 runner, R2E-Gym scaffold with its default 40-step soft budget, function calling, T=0.6, 4 parallel tasks, official swebench harness): **331/500 = 66.2%** (run 2026-08-21/22, `results/2026-08-21-deepswe38-nvfp4-tiers`). The Qwen3.6-27B run on the identical scaffold scored 69.4%; the gap is behavioural — 3.8 submits voluntarily far more often than 3.6 and explores less; giving it an 80-step budget did not help (71 vs 75/100 on the first 100 tasks — see [bench/RESULTS.md](bench/RESULTS.md#agentic-benchmarks-on-the-qwen38-daily-2026-08-2122)). Terminal-Bench 2.1 on the 2026-08-23 daily (gittensor checkpoint, terminus-2, Harbor 0.18.0, k=1, default timeouts): **50/89 = 56.2%** vs 48.3% for Qwen3.6 on the same rig — agent timeouts fell 27 to 19; TB inverts the SWE-Bench direction because its binding constraint is wall-clock and this stack decodes ~30% faster (see [bench/RESULTS.md](bench/RESULTS.md#terminal-bench-21-on-the-gittensor-daily-2026-08-23-results2026-08-23-tb21)). A control run with the most-faithful-but-2x-slower AWQ checkpoint scored 49.4% — on this benchmark serving throughput outweighs quant fidelity. The 3.6-era details are archived in [bench/RESULTS.md](bench/RESULTS.md#archive--qwen36-era-2026-07) and [docs/HISTORY.md](docs/HISTORY.md). One finding from 2026-08-21 matters for anyone re-running them: R2E-Gym's function-calling gate keys on the served model name; with it off the model never sees the tool schema and scores 5/25 instead of 20/25 on the same tasks (FINDINGS R76).

## Quick start

```bash
# 1. tier image: vLLM nightly (digest as build-arg) + LMCache 0.5.4rc4 + the rc4 patch stack
cd patches/rc4 && docker build \
  --build-arg VLLM_BASE=vllm/vllm-openai@sha256:4e9299fb10c93ba020fbbe3237f7b5998d96cfe9fae962319babc9d7796ea66e \
  -t vllm-qwen38:tiers-rc4-ba07e4a -f Dockerfile.rc4 . && cd ../..

# 2. nvfp4 KV on top of it: FA2 nvfp4 routing (PR #49891 rebase) + the sm120 V-scale overlay (AOT-built)
cd patches-nvfp4kv && docker build \
  --build-arg VLLM_BASE=vllm-qwen38:tiers-rc4-ba07e4a --build-arg VLLM_COMMIT=ba07e4a48 \
  -t vllm-qwen38:tiers-nvfp4kv -f Dockerfile.nvfp4kv . && cd ..

# 3. serve (tiers on, nvfp4 KV, V2 runner)
MODEL_DIR=/path/to/Qwen3.8-27B-MTP-NVFP4 ./scripts/serve-tier-rc4.sh
```

`serve-tier-rc4.sh` derives the LMCache chunk from the KV dtype (2864 for nvfp4, 1616 for fp8), sets `mnbt = 2·chunk − 1`, checks the pool against a band after boot, and nulls a baked tokenizer truncation if the checkpoint ships one. `KVDTYPE=fp8_e4m3 IMAGE=vllm-qwen38:tiers-rc4-ba07e4a UTIL=0.95` runs the previous (fp8) generation from the same script. Every flag is annotated inline in the script and in [docs/GOTCHAS.md](docs/GOTCHAS.md).

## Things that bite

1. The LMCache chunk must equal vLLM's unified block. vLLM logs it at boot ("Setting attention block size to N tokens"): 1616 with fp8 KV, 2864 with nvfp4. A mismatch is either a loud error or a mis-chunked cache.
2. `--no-async-scheduling` stays on with MTP. On this hybrid, async scheduling costs about 1 tool-eval point and has corrupted KV under spec decode ([vllm#42655](https://github.com/vllm-project/vllm/issues/42655)).
3. Verify the engine after launch, not the banner: image name, vLLM version, `Using V2 Model Runner`, the overlay line `NVFP4KV-SM120: linear-V-scale store overlay ACTIVE`, and the pool (about 309K). A dispatcher that printed "tier-rc4" while running a 0.23 image went unnoticed for six days.
4. The nvfp4 store overlay is required on sm120. vLLM's in-tree writer swizzles V block scales for SM100; FlashInfer's FA2 reader addresses them linearly. Behavioural probes do not catch it (needles and tool-eval pass with the wrong writer); [`patches-nvfp4kv/overlay/diag_vsf_layout.py`](patches-nvfp4kv/overlay/diag_vsf_layout.py) does: 2.7–10× the attention error without the overlay.
5. Keep util at 0.93 on the V2 runner with nvfp4 KV. At 0.95 the FlashInfer autotuner OOM-falls-back to its default tactic on the first new shape.
6. A 4-bit KV cache costs about 2 tool-eval points against fp8 on this model on the plain profile (R77/R80: 89–89.5 vs 91). The tier measurement (92 ± 1.4, n=2) did not show it; a 69×4 is pending.
7. Needle-test across a restart before trusting any external KV tier on a hybrid model. Hit counters and fluent output were compatible with a broken cache through four rounds in 2026-07 ([docs/LMCACHE.md](docs/LMCACHE.md)).

The rest: [docs/GOTCHAS.md](docs/GOTCHAS.md).

## Patch stacks

| stack | what | build |
|---|---|---|
| [patches/rc4/](patches/rc4/README.md) | LMCache 0.5.4rc4 for vLLM 0.27's hybrid layout: fused rank-4 page view (0001), strided fp8 regroup (0002), sidecar VRAM diet (0007), fs_native cap enforcement (0008) and watermark LRU eviction (0009); vLLM side: Mamba store boundary (0005), transfer-abort assert (0010). The page edits classify pages by byte accounting, which is why nvfp4 pages needed no new code. | `Dockerfile.rc4`, base digest as build-arg |
| [patches-nvfp4kv/](patches-nvfp4kv/README.md) | vLLM [PR #49891](https://github.com/vllm-project/vllm/pull/49891) (ch2lab) rebased: sm120 nvfp4 → FlashInfer FA2 routing (0001, 0001b); the linear-V-scale store overlay as a stable-ABI op (0002, 0002b); optional MTP-drafter full cudagraph (0003, unmeasured); the numeric layout diagnostic. | `Dockerfile.nvfp4kv`, FROM any vLLM image of the matching commit |

## Rejected

[docs/REJECTED.md](docs/REJECTED.md) lists every rejected configuration with the number that killed it: TurboQuant and KVarN 4-bit KV kernels (corrupt under concurrency), DFlash v1 (21K context) and DFlash2 (matched by MTP on the same runner at 40% of the pool), async scheduling, the official NVIDIA quant, and more. Read it before changing the config.

## Docs

- [docs/GOTCHAS.md](docs/GOTCHAS.md) — every flag with its failure mode.
- [docs/DESIGN.md](docs/DESIGN.md) — why W4A4 weights, where the VRAM goes, what the host contributes.
- [docs/LMCACHE.md](docs/LMCACHE.md) — the tier profile, what removing it changes, and why a tier that looks fine can be wrong.
- [docs/NVFP4KV.md](docs/NVFP4KV.md) — the NVFP4 KV cache, the overlay, the measurements.
- [docs/V2RUNNER.md](docs/V2RUNNER.md) — the V2 model runner measurement.
- [docs/DFLASH2.md](docs/DFLASH2.md) — DFlash2 vs MTP.
- [docs/HISTORY.md](docs/HISTORY.md) — daily lineage since 2026-06, reversals included.
- [bench/RESULTS.md](bench/RESULTS.md) — all tables; [bench/](bench/README.md) — probes and commands.

## License

MIT ([LICENSE](LICENSE)) for the original work here. Anything derived from vLLM, LMCache or FlashInfer — redistributed PR diffs, patch context, patched files inside built images — stays Apache-2.0-derived; per-file inventory in [THIRD_PARTY.md](THIRD_PARTY.md).

## Credits

- [vLLM](https://github.com/vllm-project/vllm), [LMCache](https://github.com/LMCache/LMCache), [FlashInfer](https://github.com/flashinfer-ai/flashinfer).
- ch2lab for [vLLM PR #49891](https://github.com/vllm-project/vllm/pull/49891) (sm120 nvfp4 KV routing).
- drowzeys, whose [DGX Spark repo](https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-27B-ADay777Ablit-NVFP4-A4Q-NVFP4-KV-4M-KV-token-pool-MTP3-Single-DGX-Spark) carried the linear-V-scale writer patch this stack adopts.
- [sakamakismile](https://huggingface.co/sakamakismile/Qwen3.8-27B-MTP-NVFP4) for the W4A4 export.
- z-lab and inco.ai for the DFlash2 drafter and [vLLM PR #52816](https://github.com/vllm-project/vllm/pull/52816), measured and rejected here.
