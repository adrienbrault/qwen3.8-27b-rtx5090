# Every flag, and why

The serve command lives in [`../scripts/serve.sh`](../scripts/serve.sh). This explains it.

## Model

`unsloth/Qwen3.6-27B-NVFP4` — compressed-tensors NVFP4 (4-bit weights), with a **vision tower** and an **MTP head**. ~22 GB on disk.

Chosen over the alternatives by measurement:

| quant | why not |
|---|---|
| `nvidia/Qwen3.6-27B-NVFP4` (official) | ~2.6× slower prefill (4.9K vs 12.4K t/s @4K), 20–25% slower decode, OOMs at util 0.95, MTP crashes at moderate batch. |
| natfii NVFP4 (modelopt) | Faster prefill (13.3K), but **lost Terminal-Bench 2.1** to Unsloth (12/16 vs 15/16 trials, 7/8 vs 8/8 pass@2). Quality beat speed. |
| Intel AutoRound int4 | Faster decode, ~4× slower prefill. Good if your workload is output-heavy, bad for big-context coding. |

> **Loading note:** Unsloth's build is compressed-tensors. Pass **no** `--quantization` flag (auto-detects). Passing `modelopt` errors out.

## The flags

```bash
--kv-cache-dtype turboquant_4bit_nc --block-size 128
```
The whole point. 4-bit KV → **47.8K tokens/GiB vs fp8's 26.8K** (1.8× denser) → 261K pool instead of 172K. `block-size 128` is required by the TurboQuant kernel. **Needs the patched image** — on stock vLLM this + MTP produces garbage.

```bash
-e VLLM_TQ_PRESET=turboquant_4bit_nc      # REQUIRED
```
Not optional. vLLM never propagates the KV dtype to the MTP **draft** runner (it arrives as `"auto"` and crashes). `tq_auto_fallback.py` reads this env as the fallback. Must match `--kv-cache-dtype`.

```bash
--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
```
MTP speculative decoding. `ns=3` is the sweet spot (ns=4 is unstable, ns=1/2 leave speed on the table). **Costs ~0 VRAM** — the draft head ships inside the model weights. Acceptance ~76%.

⚠️ Known to **crash at 16+ concurrent** (upstream vLLM MTP bug). Drop `--speculative-config` entirely if you need heavy concurrency.

```bash
--gpu-memory-utilization 0.94 --max-model-len 150000
```
Let vLLM *profile* the memory. Do **not** hand-set `--kv-cache-memory` to the "fully utilize" value vLLM suggests in its logs — that hint ignores warmup transients and OOMs.

**Why max-len 150K, not "whatever the pool holds":** TurboQuant's `_continuation_prefill`
dequantizes the entire cached prefix to bf16 (~4 KB/token of transient scratch). Past ~160K of
cached context that allocation exceeds the activation headroom and the **engine dies mid-prefill**
(`torch.OutOfMemoryError` in `k_full[:cached_len] = ...`). 147K is verified end-to-end. The pool
above 150K still earns its keep: each concurrent sequence pins a fixed GDN/Mamba state
(~30K-token-equivalent), so concurrency eats pool fast.

**Why 0.94, not 0.95:** at 0.95 (261K pool) a burst of ~8 simultaneous cold prompts **crashes the
engine** — the GDN prefill kernel (`chunk_fwd_o`) needs a ~96 MiB transient workspace per step, and
with 6 requests' prefills packed into one 8192-token batch there is no headroom left
(`torch.OutOfMemoryError`, engine dead, container restart). `expandable_segments:True` was already
on; it doesn't save you. 0.94 gives back ~320 MiB of activation headroom (222,580-token pool at
max-len 150K, measured). Verified: the same 8-cold-prompt burst that killed 0.95 passes cleanly
at 0.94. If you never see concurrent cold starts (single-user, one client), 0.95 works and buys
a bit more pool.

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
