# DFlash2 instead of MTP — measured on the 5090 (2026-08-21)

**Status: measured, not the daily.** [DFlash2](https://inco.ai/blog/dflash2/) (block-parallel drafter + candidate selector + local convolutions, vLLM [PR #52816](https://github.com/vllm-project/vllm/pull/52816), merged 2026-08-21) is the fastest single-stream decode ever measured on this box and lossless on tool-eval — and it costs two thirds of the context window on a 32 GB card. Full detail in the repo's FINDINGS R78.

## Stack

- Base: nightly `ba07e4a48` (the newest nightly image, cut 100 minutes *before* the PR merged) + the PR as a Python/Triton patch — [`patches-dflash2/`](../patches-dflash2/) (Dockerfile + diff). No kernel build.
- Drafter: [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (= z-lab's), 3.85 GB bf16, 5 layers tapping target layers 5/19/33/47/61, SWA-2048, block 8.
- Target: the daily's saka W4A4 checkpoint, fp8 KV, plain profile. Launcher: [`scripts/serve-dflash2.sh`](../scripts/serve-dflash2.sh).
- Gotchas that bite: `method` is `dflash` (the 2 comes from the draft's architecture); `num_speculative_tokens` **must be 7**; `--async-scheduling` is refused; greedy is unsupported by the selector; it **forces the V2 model runner**; the drafter does not receive multimodal embeddings (vision requests draft blind); the merged PR reads the target LM head through `LogitsProcessor`, so an unquantized-but-compressed-tensors `lm_head` works without #52883.

## Numbers (RTX 5090, Qwen3.8-27B saka W4A4 + fp8 KV, T=0.6, effort medium, llama-benchy)

| | DFlash2 n=7 | autoregressive (same image) | MTP ns=4 (fp8 plain, V1 runner) |
|---|---|---|---|
| decode c1, pp8192 | **164.5 t/s** (peak 239) | 73.2 | 123.1 |
| decode c1, pp30000 | **170.4** (flat) | 71.4 | — |
| decode c2 / c4 | **298.6** / 267.0 | 126.4 / 213.4 | 207 / 315 |
| decode c8 (pp4096) | 294.9 | — | — |
| prefill c1 8K / 32K / 55K | 13.5–14.4K / 10.4K / 8.3K | 13.9K | 13.0K / 9.7K |
| prefill aggregate c4 / c8 | 5.7K / 2.7K (TTFT 2.8 s ± 2.0 / 5.5 s ± 3.7) | 14.8K (c4) | 12.5K (c4) |
| acceptance | 2.60 accepted per draft → ~3.6 tokens/step | — | ~2.3–2.6 |
| tool-eval 69×2 | **91 ± 1.4** | — | 91 |
| depth needles 9K/20K/40K, cold+warm | 6/6 + 6/6 | — | — |
| **max context @ util 0.95–0.96** | **~62K** (pool 66,419 @60K) | 232,000 @60K | 200K (pool 207K) |

**2.25× over autoregressive, +34% over MTP single-stream, flat with depth** — and at parity on tool-eval. The acceptance (3.6 tokens/step) is well under the blog's 4.8, which was GSM8K at T=1.0/xhigh; benchy's synthetic prompts and T=0.6 explain most of that.

## Why it is not the daily

1. **Context.** At max-len 200K the engine refuses to start (3.9 GiB left for KV, `estimated maximum model length is 62624`). Without the drafter the same image at 60K has a 232K-token pool; with it, 66K — the DFlash2 configuration costs **~166K fp8-KV tokens (~7.8 GiB)**, far more than its 3.85 GB of weights: the draft's KV is sized for the full max-len despite its 2048-token sliding window, plus the V2 speculator's buffers. Same class of wall DFlash v1 hit here in July (21K), at a higher number.
2. **Concurrency.** c4 decode is 15% below MTP and concurrent prefill collapses (5.7K t/s at c4, 2.7K at c8) — this box serves parallel agents.
3. **Untested surface**: LMCache tiers on the V2 runner, vision with a blind drafter, and a PR that is one day old.

It is an excellent *second profile* for a single user under 60K of context, and it becomes a daily candidate the day the drafter's KV is allocated by its sliding window.

**On the +34%:** the MTP column is the MTP we actually serve (V1 runner, nightly `ac7509e2b`). The same-runner control — MTP on the V2 runner with this image — could not run: the `ba07e4a48` nightly's `_C_stable_libtorch.abi3.so` raises an undefined cutlass symbol (`GemmUniversalBase…enable_sm89_to_sm90`) on that path. The 2.25× over autoregressive is same-image, same-runner.
