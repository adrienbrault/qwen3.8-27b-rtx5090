# Every flag, and why

> ⚠️ **STALE — this documents the 2026-07-15 TurboQuant daily, which is retired.** TurboQuant KV was closed permanently for this hybrid (needle-wrong at 60K, dead under the concurrency killer — see [HISTORY.md](HISTORY.md)). The current daily is natfii NVFP4 W4A4 + fp8 KV + MTP `ns=4` + **LMCache DRAM/NVMe tiers**: flags in [`../scripts/serve.sh`](../scripts/serve.sh) and [GOTCHAS.md](GOTCHAS.md#config-essentials), tier specifics in [LMCACHE.md](LMCACHE.md), no-tiers variant in [`../scripts/serve-plain.sh`](../scripts/serve-plain.sh). Kept for the flag rationale that's still shared (parsers, sampling, cache mounts, host notes) and as the record of the TurboQuant era.

## Recommended daily: turboquant_4bit_nc on the clean TQ image

**As of 2026-07-15 this is what we run in production.** The daily is the patched TurboQuant image with **`turboquant_4bit_nc`** KV (4-bit MSE Keys / 4-bit Values + norm-correction) **plus `--no-async-scheduling`**: a **~235K-token pool → 200K usable context** (**+42% pool, +25% context** over the earlier `turboquant_k8v4` daily), at a small single-stream decode cost. `k8v4` (8-bit K / 4-bit V) stays a decode-optimal [middle-ground alternative](#alternative-clean-tq-image--turboquant_k8v4), and fp8 KV stays the [stock alternative](#alternative-stock-nightly--fp8-kv) for deep-context high-concurrency batch serving.

**Why `4bit_nc` is back — and why the flag.** This repo previously *rejected* `turboquant_4bit_nc` ("4-bit keys destroy retrieval, 0/8"). That was a **confound**: vLLM's async scheduler desyncs its request-ID→batch-row mapping under MTP's multi-token verify batches ([vllm#42655](https://github.com/vllm-project/vllm/issues/42655)) and corrupts KV — catastrophically for 4-bit keys, subtly (~10%) for 8-bit. `--no-async-scheduling` removes it; genuine `4bit_nc` then scores **8/8 @9K/20K/40K and 90/90 under high-pressure concurrency**. See the [reversal](HISTORY.md#status-turboquant_4bit_nc-is-the-daily-the-asyncspec-reversal).

The flags that make `4bit_nc` the daily (everything else — MTP, parsers, sampling, caches, host notes — is shared with the alternatives and detailed below):

```bash
--kv-cache-dtype turboquant_4bit_nc                     # 4-bit K / 4-bit V + norm-correction; needs the patched image
-e VLLM_TQ_PRESET=turboquant_4bit_nc                    # MUST equal --kv-cache-dtype
--no-async-scheduling                                   # CRITICAL — see below; without it 4bit_nc+MTP corrupts KV (0/8)
--gpu-memory-utilization 0.94 --max-model-len 200000    # ~235K-token pool; block auto-resolves (don't hand-set)
--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
--structured-outputs-config '{"backend":"xgrammar","reasoning_parser":"qwen3","enable_in_reasoning":false}'
--default-chat-template-kwargs '{"preserve_thinking":true}'
--mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill
--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
```

decode @512 tg-mean **c1 133, c2 211, c4 432, c8 435** (−3% c1, −7% c8 vs k8v4); MTP mean acceptance length ~3.2. Needle-in-haystack **8/8 @9K, 8/8 @20K, 8/8 @40K**; **90/90** under high-pressure concurrency — with `--no-async-scheduling`. tool-eval-bench 89 (with the #44993 SO graft). Full numbers in [../bench/RESULTS.md](../bench/RESULTS.md).

**Verify container identity after launch.** Confirm the startup log reports a **~235K-token KV pool**, not ~165K. A silent fallback to `turboquant_k8v4` (preset/dtype mismatch, or the wrong image) reads as a perfectly healthy server at the *wrong* config — you quietly get the smaller pool. The pool size in the log is the fastest tell.

## Model

`unsloth/Qwen3.6-27B-NVFP4` — compressed-tensors NVFP4 (4-bit weights), with a **vision tower** and an **MTP head**. ~22 GB on disk.

Chosen over the alternatives by measurement:

| quant | why not |
|---|---|
| `nvidia/Qwen3.6-27B-NVFP4` (official) | ~2.6× slower prefill (4.9K vs 12.4K t/s @4K), 20–25% slower decode, OOMs at util 0.95, MTP crashes at moderate batch. |
| natfii NVFP4 (modelopt) | Faster prefill (13.3K), but **lost a 16-task Terminal-Bench 2.1 probe** to Unsloth (12/16 vs 15/16 trials, 7/8 vs 8/8 pass@2) — an internal model-selection spot-check, not the official 89-task benchmark (that result, on the later natfii daily, is [in the README](../README.md#agentic-benchmark-results)). At the time, quality beat speed. (natfii later became the daily anyway, on the strength of the full 69×2 tool-eval — see [HISTORY.md](HISTORY.md).) |
| Intel AutoRound int4 | Faster decode, ~4× slower prefill. Good if your workload is output-heavy, bad for big-context coding. |

> **Loading note:** Unsloth's build is compressed-tensors. Pass **no** `--quantization` flag (auto-detects). Passing `modelopt` errors out.

## The flags (turboquant_4bit_nc daily)

> These are the flags for the patched TurboQuant image — **our daily** as of 2026-07-15. The `--kv-cache-dtype`, `-e VLLM_TQ_*`, `--no-async-scheduling`, `--max-model-len`, `--structured-outputs-config`, and `--default-chat-template-kwargs` values below are the ones that differ from the [alternatives](#alternative-clean-tq-image--turboquant_k8v4); the rest apply to all.

```bash
--kv-cache-dtype turboquant_4bit_nc
```
The whole point. **4-bit Keys (MSE) / 4-bit Values + norm-correction** → **~48K tokens/GiB** → a **~235,000-token pool** in ~the same KV memory k8v4 uses for 165K (~4.89 GiB). **Needs the patched image** — on stock vLLM this + MTP produces garbage. Do **not** hand-set `--block-size`: the hybrid allocator auto-resolves the unified block for mamba-page parity. 4-bit keys are more retrieval-sensitive to scheduler corruption than 8-bit — which is why **`--no-async-scheduling` is mandatory** (below), and why this preset was mis-rejected before the async fix.

```bash
-e VLLM_TQ_PRESET=turboquant_4bit_nc      # REQUIRED — must equal --kv-cache-dtype
```
Not optional. vLLM never propagates the KV dtype to the MTP **draft** runner (it arrives as `"auto"` and crashes). `tq_auto_fallback.py` reads this env as the fallback. **Must equal `--kv-cache-dtype` exactly** — a mismatch puts the draft path on a different KV format than the target, and a mismatch to `turboquant_k8v4` is the silent-fallback trap: the server comes up healthy at the smaller 165K pool. Check the logged pool size (~235K).

```bash
--no-async-scheduling                     # CRITICAL for turboquant_4bit_nc + MTP
```
**The flag that un-rejected `4bit_nc`.** vLLM's async scheduler emits its request-ID→batch-row mapping one step ahead; under MTP, every step is a multi-token *verify* batch, so even a **single** request spans multiple batch rows and the async mapping desyncs ([vllm#42655](https://github.com/vllm-project/vllm/issues/42655)) → KV written to the wrong slots → corruption. 4-bit keys are far more sensitive to it (**0/8** retrieval) than 8-bit (~10% intermittent). With this flag, genuine `4bit_nc` is clean: **8/8 @9K/20K/40K, 90/90 high-pressure concurrent.** Broader lesson: the "all 4-bit-KV kernels corrupt under concurrency" reputation was very likely this scheduler bug, not the kernels.

```bash
--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
```
MTP speculative decoding. `ns=3` is the sweet spot (ns=4 is unstable, ns=1/2 leave speed on the table). **Costs ~0 VRAM** — the draft head ships inside the model weights. Acceptance ~76%.

⚠️ Known to **crash at 16+ concurrent** (upstream vLLM MTP bug). Drop `--speculative-config` entirely if you need heavy concurrency.

```bash
--gpu-memory-utilization 0.94 --max-model-len 200000
```
Let vLLM *profile* the memory. Do **not** hand-set `--kv-cache-memory` to the "fully utilize" value vLLM suggests in its logs — that hint ignores warmup transients and OOMs.

**Why max-len 200K, not "whatever the pool holds":** TurboQuant's `_continuation_prefill`
dequantizes the entire cached prefix to bf16 (~4 KB/token of transient scratch). Far past the cap
that allocation exceeds the activation headroom and the **engine dies mid-prefill**
(`torch.OutOfMemoryError` in `k_full[:cached_len] = ...`). The ~235K-token pool above the cap
still earns its keep: each concurrent sequence pins a fixed GDN/Mamba state, so concurrency eats
pool fast.

```bash
--structured-outputs-config '{"backend":"xgrammar","reasoning_parser":"qwen3","enable_in_reasoning":false}'
```
Enables the **reasoning gate** the [#44993 graft](../patches/README.md) needs. Without it, `response_format` json_schema with thinking-on returned **empty `content`** (the schema JSON leaked into `reasoning_content` because the grammar never re-engaged after `</think>` when MTP rejected drafts). `enable_in_reasoning:false` holds the grammar off *inside* the think block and re-arms it for the answer. Lifted tool-eval **85→89**. **Requires an adequate `max_tokens`** — reasoning + JSON both have to fit; too small a budget truncates mid-think and still looks empty.

```bash
--default-chat-template-kwargs '{"preserve_thinking":true}'
```
Keeps historical `<think>` blocks in the rendered prompt across turns. **Client caveat:** for this to actually persist, the client must resend prior reasoning in the **`reasoning`** message field — **not** the deprecated `reasoning_content`, which vLLM ignores on *input*. Send it in the wrong field and the template silently has nothing to preserve.

**Why 0.94, not 0.95:** at 0.95 a burst of ~8 simultaneous cold prompts **crashes the
engine** — the GDN prefill kernel (`chunk_fwd_o`) needs a ~96 MiB transient workspace per step, and
with several requests' prefills packed into one 8192-token batch there is no headroom left
(`torch.OutOfMemoryError`, engine dead, container restart). `expandable_segments:True` was already
on; it doesn't save you. 0.94 gives back enough activation headroom that the same 8-cold-prompt
burst passes cleanly. If you never see concurrent cold starts (single-user, one client), 0.95
works and buys a bit more pool.

```bash
--max-num-seqs 8
```
Fewer sequence slots → less activation memory → bigger KV pool. It does **not** partition the KV cache (that's a shared paged pool). 8 is a good balance; `1` would buy a little more context but serializes everything.

```bash
--max-num-batched-tokens 8192
```
**Do not lower this.** 4096 costs a **~4× prefill regression** (9.6K → 2.6K t/s) to buy +28K context. Bad trade, tested.

```bash
--mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill
```
Qwen3.6 is a hybrid (GDN/linear-attention + full-attention). `align` mode is what lets prefix caching work on the Mamba layers. Prefix caching gives a real **5× speedup on repeated prompts** — even though vLLM's own metric claims 0% (see README gotchas).

```bash
--limit-mm-per-prompt '{"image":4,"video":0}'
```
Vision on. Costs roughly 60K tokens of context. Drop it (+ `--language-model-only`) if you want a text-only mega-context mode.

```bash
--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
```
The model emits XML-format tool calls. `qwen3_xml` is correct; `hermes` silently drops them.

```bash
--override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}'
```
Qwen's recommended sampling. (Their *benchmark* config uses `temperature: 1.0` — match that if you're reproducing published scores.)

## Alternative: clean TQ image + turboquant_k8v4

`turboquant_k8v4` (**8-bit Keys / 4-bit Values**) was the daily before `4bit_nc`, and stays the **decode-optimal middle ground**: it costs ~3–7% less decode than `4bit_nc` short-context (and up to +13% at deep single-stream) but carries a smaller pool — **165,274 tokens → 160K context, 33.8K tok/GiB** (vs `4bit_nc`'s ~235K / ~48K). Pick it if you want the last few percent of decode and don't need the extra ~40K of context. It runs on the same patched image; **still use `--no-async-scheduling`** (8-bit keys hide the async×spec corruption as only ~10% intermittent degradation, but it's the same bug). Only the KV-dtype/preset and context flags differ from the daily:

```bash
--kv-cache-dtype turboquant_k8v4
-e VLLM_TQ_PRESET=turboquant_k8v4                      # MUST equal --kv-cache-dtype
--no-async-scheduling
--gpu-memory-utilization 0.94 --max-model-len 160000   # 165,274-token pool
```

decode @512 tg-mean **c1 137, c2 250, c4 426, c8 467** (fresh same-session, async-off); needle-in-haystack **8/8 @9K, 8/8 @20K, 6/6 @40K**; tool-eval **89**. *(An earlier standalone bench quoted k8v4 at c1 164 @512; a fresh same-session re-measurement did not reproduce it — ~137 is the reproduced number. See [../bench/RESULTS.md](../bench/RESULTS.md).)*

## Alternative: stock nightly + fp8 KV

fp8 KV on **plain `vllm/vllm-openai:nightly`** (no patched image) is the pick for **deep-context high-concurrency batch serving** — the one regime it beats k8v4 (decode c4@4096: fp8 461 vs k8v4 277 t/s) — and the most battle-tested path. It gives up single-stream speed and ~29K of pool. Only the KV-cache and context flags differ from the daily above; everything else (MTP, parsers, sampling, caches, host notes) is identical.

```bash
--kv-cache-dtype fp8_e4m3
--gpu-memory-utilization 0.94 --max-model-len 131072   # fp8 fits 136,477 tokens @ 0.94; 131072 leaves headroom
--max-num-seqs 8 --max-num-batched-tokens 8192
--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
```

decode @512 tg-mean **c1 130, c2 251, c4 482, c8 478** (peak c8 832); prefill @4K **9,604** t/s; deep-context (`pp=4096`) it leads from c2 up. MTP mean acceptance length ~3.2, same as k8v4.

## Environment

```bash
-e PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512   # fragmentation
-e TORCH_MATMUL_PRECISION=high
```

Mount the compile caches on **every** launch — content-addressed, shared across models, turns a ~200s cold start into ~120s:

```bash
-v .../cache/torch_compile:/root/.cache/vllm/torch_compile_cache
-v .../cache/triton:/root/.triton/cache
-v .../cache/inductor:/root/.cache/inductor
```

## Host notes

**GPU overclock (affects the benchmark numbers).** This box runs a **memory-only overclock: +4500 MHz VRAM offset** (16,051 MHz effective, vs ~14,001 stock) at the 600 W power limit, persisted across reboots via a systemd unit. Core clock is left at stock (0 offset) — deliberately, because decode throughput on this model is **memory-bandwidth-bound**, so the memory OC is the only knob that moves the needle and a core OC would add heat for nothing. **All throughput numbers in [../bench/RESULTS.md](../bench/RESULTS.md) are with this OC.** A stock 5090 will decode somewhat slower; prefill (compute-bound) is barely affected. Reproduce with:

```bash
sudo nvidia-smi -pm 1 && sudo nvidia-smi -pl 600 && sudo nvidia-smi -rgc
sudo python3 -c "import pynvml as N; N.nvmlInit(); h=N.nvmlDeviceGetHandleByIndex(0); \
  N.nvmlDeviceSetGpcClkVfOffset(h,0); N.nvmlDeviceSetMemClkVfOffset(h,4500)"
```
Validate the offset actually applied (a "running" service isn't proof): `nvidia-smi -q -d CLOCK`, or read `nvmlDeviceGetMemClkVfOffset`. +4500 is stable on this sample; your mileage varies — step up and watch for memory-ECC errors or artifacts.

**Disable swap.** Over-committing RAM (e.g. running a benchmark container next to vLLM) swap-thrashes the box into a hard hang instead of failing fast. `swapoff -a` turns a wedge into a clean OOM kill.
