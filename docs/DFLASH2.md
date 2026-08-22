# DFlash2 instead of MTP — measured on the 5090 (2026-08-21)

**Status: measured, rejected for this box.** [DFlash2](https://inco.ai/blog/dflash2/) (block-parallel drafter + candidate selector + local convolutions, vLLM [PR #52816](https://github.com/vllm-project/vllm/pull/52816), merged 2026-08-21) decodes at 164 t/s single-stream and is lossless on tool-eval — but so does MTP on the same runner (160 t/s), and DFlash2 pays for its 3% with two thirds of the context window and −30% at c4. The +34% headline against our served MTP was the V2 model runner / newer nightly, not the drafter. Full detail in the repo's FINDINGS R78.

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

**The same-runner control changes the story.** MTP ns=4 on the V2 runner with this very image (60K/0.96): **c1 160.4 t/s, c2 251, c4 380, deep-30K 156, pool 166,071**. So against the MTP we actually serve (V1 runner, older nightly: 123 t/s) DFlash2 looks +34% — but against MTP on the *same* runner it is **+3% single-stream (noise), +19% at c2, −30% at c4, with 40% of the KV pool**. The +30% was the V2 model runner and/or the 276 newer commits, not the drafter; MTP's own uplift over AR on V2 is 2.19× vs DFlash2's 2.25×. (First attempt at this control died on a flaky spawn-child `ImportError: undefined symbol …cutlass…GemmUniversalBase…` in the nightly's stable-ABI extension; the retry booted clean.)

**What to pursue instead**: the tier daily (fp8 KV + LMCache tiers + MTP) on the V2 runner / newer nightly. That is where the 123 → 160 t/s lives, with the full 200K context.

## Quantized drafters (patch 0002, 2026-08-22)

The HF quantized drafters (`syvai/Qwen3.8-27B-DFlash2-W4A16` 1.2 GB, `lued/Qwen3.8-27B-DFlash2-W8` 2.0 GB) failed to load on upstream #52816 with `'QKVParallelLinear' object has no attribute 'weight'`. Upstream already threads `get_draft_quant_config()` into the drafter's layers; the one break is `DFlashQwen3Model._build_context_kv_buffers`, which reads `qkv_proj.weight[q_size:]` densely to precompute context KV in a single fused GEMM — on a compressed-tensors pack-quantized layer that is `weight_packed` + `weight_scale`. `patches-dflash2/0002-quantized-drafter-kv-precompute.diff` ports syv-ai's `_dense_kv_rows` ([qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) `patches/dflash2-backport.patch`): the buffers are built at the end of `load_weights`, before `process_weights_after_loading`'s Marlin repack, so the packed tensors are still in checkpoint layout and dequantize there (symmetric, group strategy). The backport's quantized-lm_head hunk is not needed here (our targets keep `lm_head` unquantized). `test_dense_kv_rows.py` is a CPU round-trip that runs inside the image and is bit-exact for W4 and W8. Measurements of the quantized drafters: pending (R89).
