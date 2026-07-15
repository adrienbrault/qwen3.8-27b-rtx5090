# Every flag, and why

The serve command lives in [`../scripts/serve.sh`](../scripts/serve.sh). This explains it.

## Recommended: clean TQ image + turboquant_k8v4 (our daily)

**As of 2026-07-15 this is what we run in production.** The daily is the patched TurboQuant image with **`turboquant_k8v4`** KV (8-bit Keys / 4-bit Values): faster single-stream than fp8, **+21% pool** (165K vs 136K tokens, in *less* KV memory), and equal long-context retrieval. It replaces our earlier fp8 daily — the "TurboQuant corrupts" call was a misdiagnosis (a noisy soak-test detector + 4-bit-*key* quality loss), fixed by keeping keys at 8 bits. See the [status note](../README.md#status-turboquant_k8v4-is-the-daily). fp8 KV stays the [documented alternative](#alternative-stock-nightly--fp8-kv) for deep-context high-concurrency batch serving.

The flags that make k8v4 the daily (everything else — MTP, parsers, sampling, caches, host notes — is shared with the fp8 alternative and detailed below):

```bash
--kv-cache-dtype turboquant_k8v4                        # 8-bit K / 4-bit V; needs the patched image
-e VLLM_TQ_PRESET=turboquant_k8v4                       # MUST equal --kv-cache-dtype
--gpu-memory-utilization 0.94 --max-model-len 160000    # 165,274-token pool; block auto-resolves to 2112
--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
--mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill
--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
```

decode @512 tg-mean **c1 164, c2 319, c4 524, c8 516**; prefill @4K **10,392** t/s; MTP mean acceptance length ~3.2. Needle-in-haystack **8/8 @9K, 8/8 @20K, 6/6 @40K** — matches fp8. Full numbers in [../bench/RESULTS.md](../bench/RESULTS.md).

## Model

`unsloth/Qwen3.6-27B-NVFP4` — compressed-tensors NVFP4 (4-bit weights), with a **vision tower** and an **MTP head**. ~22 GB on disk.

Chosen over the alternatives by measurement:

| quant | why not |
|---|---|
| `nvidia/Qwen3.6-27B-NVFP4` (official) | ~2.6× slower prefill (4.9K vs 12.4K t/s @4K), 20–25% slower decode, OOMs at util 0.95, MTP crashes at moderate batch. |
| natfii NVFP4 (modelopt) | Faster prefill (13.3K), but **lost Terminal-Bench 2.1** to Unsloth (12/16 vs 15/16 trials, 7/8 vs 8/8 pass@2). Quality beat speed. |
| Intel AutoRound int4 | Faster decode, ~4× slower prefill. Good if your workload is output-heavy, bad for big-context coding. |

> **Loading note:** Unsloth's build is compressed-tensors. Pass **no** `--quantization` flag (auto-detects). Passing `modelopt` errors out.

## The flags (turboquant_k8v4 daily)

> These are the flags for the patched TurboQuant image — **our daily** as of 2026-07-15. The `--kv-cache-dtype`, `-e VLLM_TQ_*`, and `--max-model-len` values below are the ones that differ from the [fp8 alternative](#alternative-stock-nightly--fp8-kv); the rest apply to both.

```bash
--kv-cache-dtype turboquant_k8v4
```
The whole point. **8-bit Keys / 4-bit Values** → **33.8K tokens/GiB vs fp8's 26.0K** → a **165,274-token pool** in *less* KV memory (4.89 GiB vs fp8's 5.25). The 8-bit keys are what preserve long-context retrieval — the 4-bit-key `turboquant_4bit_nc` variant scored **0/8** on fair needle-in-haystack and is rejected. **Needs the patched image** — on stock vLLM this + MTP produces garbage. Do **not** hand-set `--block-size`: the hybrid allocator auto-resolves the unified block to **2112** (vs fp8's 1600), adding 3 padding layers (up to 6.25% KV waste) for mamba-page parity.

```bash
-e VLLM_TQ_PRESET=turboquant_k8v4         # REQUIRED — must equal --kv-cache-dtype
```
Not optional. vLLM never propagates the KV dtype to the MTP **draft** runner (it arrives as `"auto"` and crashes). `tq_auto_fallback.py` reads this env as the fallback. **Must equal `--kv-cache-dtype` exactly** — a mismatch puts the draft path on a different KV format than the target.

```bash
--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
```
MTP speculative decoding. `ns=3` is the sweet spot (ns=4 is unstable, ns=1/2 leave speed on the table). **Costs ~0 VRAM** — the draft head ships inside the model weights. Acceptance ~76%.

⚠️ Known to **crash at 16+ concurrent** (upstream vLLM MTP bug). Drop `--speculative-config` entirely if you need heavy concurrency.

```bash
--gpu-memory-utilization 0.94 --max-model-len 160000
```
Let vLLM *profile* the memory. Do **not** hand-set `--kv-cache-memory` to the "fully utilize" value vLLM suggests in its logs — that hint ignores warmup transients and OOMs.

**Why max-len 160K, not "whatever the pool holds":** TurboQuant's `_continuation_prefill`
dequantizes the entire cached prefix to bf16 (~4 KB/token of transient scratch). Far past the cap
that allocation exceeds the activation headroom and the **engine dies mid-prefill**
(`torch.OutOfMemoryError` in `k_full[:cached_len] = ...`). The 165,274-token pool above the cap
still earns its keep: each concurrent sequence pins a fixed GDN/Mamba state, so concurrency eats
pool fast.

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
