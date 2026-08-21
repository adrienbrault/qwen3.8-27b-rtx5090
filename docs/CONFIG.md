# Every flag, and why (daily of 2026-08-21)

The launcher is [`../scripts/serve-tier-rc4.sh`](../scripts/serve-tier-rc4.sh); its inline comments are canonical. This page gives the reason and the failure mode for each flag. Previous generations (TurboQuant, AutoRound, the 0.23 base) are in [HISTORY.md](HISTORY.md).

## Container

```bash
-e VLLM_USE_V2_MODEL_RUNNER=1
```
The V2 GPU model runner. Measured 2026-08-21 on the fp8 tier profile against the V1 runner the same hour: decode c1 128 → 152, c4 309 → 360, 30K-deep c1 119 → 143 t/s ([V2RUNNER.md](V2RUNNER.md)). The MTP cliff at ≥50K prompt tokens in flight with nvfp4 KV (R77) is gone on it. Known wart on nightly `ba07e4a48`: one boot in six raised an `ImportError` (undefined cutlass symbol in `_C_stable_libtorch`) in the spawned engine-core child; a retry boots clean. At util 0.98 on the plain profile the first request OOMs inside the FlashInfer `fp4_gemm` autotuner; keep util at 0.95 (fp8) or 0.93 (nvfp4).

```bash
-e VLLM_ATTENTION_BACKEND=FLASHINFER
-e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728
```
FlashInfer is the only backend with the FA2 nvfp4 KV path on sm120. The workspace cap (128 MiB) bounds the lazily allocated autotune buffer; perf-neutral in every regime measured.

```bash
-e LMCACHE_MP_GPU_STAGING_BATCH_SIZE=1 -e CUDA_MODULE_LOADING=LAZY
```
The sidecar's VRAM diet: 1,412 → 796 MiB, which is what buys the tier profile its utilization.

```bash
-e TORCHINDUCTOR_COMPILE_THREADS=8 -e MAX_JOBS=4 -e FLASHINFER_NUM_COMPILE_JOBS=4
-v .../cache/torch_compile_<profile>:/root/.cache/vllm/torch_compile_cache
-v .../cache/triton:/root/.triton/cache -v .../cache/inductor:/root/.cache/inductor
-v .../cache/flashinfer:/root/.cache/flashinfer
```
Cap the JIT parallelism and persist the caches. Unbounded `nvcc` on the first forward has livelocked the host (64 GB RAM). One torch.compile cache per KV dtype: the captured graphs differ.

`--ipc=host --entrypoint bash`: CUDA-IPC between engine and sidecar needs host IPC; the image entrypoint is `vllm serve` and would swallow the `bash -c`. Never `PYTORCH_ALLOC_CONF=expandable_segments`: cuMem/VMM memory is not CUDA-IPC-exportable ([pytorch#165685](https://github.com/pytorch/pytorch/issues/165685), [vllm#29544](https://github.com/vllm-project/vllm/issues/29544)) and `register_kv_caches` silently times out after 300 s.

## Sidecar

```bash
lmcache server --host 0.0.0.0 --port 5555 --chunk-size 2864 \
  --l1-size-gb 24 --l1-init-size-gb 2 --eviction-policy LRU --worker-reap-timeout-seconds 0 \
  --l2-adapter '{"type":"fs_native","base_path":"/l2","max_capacity_gb":200,"num_workers":4,"eviction":{"eviction_policy":"LRU","trigger_watermark":0.8,"eviction_ratio":0.2}}'
```
- `--chunk-size` must equal vLLM's unified hybrid block, which vLLM logs at boot ("Setting attention block size to N tokens to ensure that attention page size is >= mamba page size"): 2864 with nvfp4 KV, 1616 with fp8 KV (1568 without MTP). The launcher derives it from `KVDTYPE`.
- `--l1-size-gb 24`: pinned host RAM; `drop_caches` first. It must exceed the hot working set / 0.8 or LRU evicts the oldest session's head chunks first and the hit rate goes to 0%.
- `--worker-reap-timeout-seconds 0`: the default reaper (120 s + a lazily started heartbeat) reaps the worker after one long idle span and the cache becomes a zombie (`found_count=0`, stores dropped).
- The `eviction` block is what evicts; patches 0008/0009 make `max_capacity_gb` enforced and add watermark LRU. Unpatched, L2 grew to 876 GB against a 60 GB cap. Use a fresh L2 directory per stack generation: on-disk page format and namespace change with the KV dtype and the LMCache build.

## Engine

```bash
--kv-cache-dtype nvfp4
```
NVFP4 KV: E2M1 values, one FP8 E4M3 scale per 16 elements, 0.5625 B/element. Pool 309,090 at util 0.93 vs 208,450 for `fp8_e4m3` ([NVFP4KV.md](NVFP4KV.md)). Requires the nvfp4kv patch stack on sm120: PR #49891's FA2 routing and the linear-V-scale store overlay. Without the overlay the writer emits SM100-swizzled V scales that the FA2 reader addresses linearly; behavioural probes pass anyway and the attention error is 2.7–10× higher. `nvfp4_4over6` (scale chosen so the block max maps to 4) measured no better (tool-eval 88.5 vs 89 on the plain profile, R80).

```bash
--kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'
```
The only connector that handles this hybrid's opaque GDN state pages. On the V2 runner it attaches as-is; the tier correctness gates (needles cold/warm, restart-proof revisit, concurrent loaders) passed on both fp8 and nvfp4 pages on 2026-08-21.

```bash
--max-num-batched-tokens 5727          # = 2·chunk − 1 (3231 with fp8 KV)
--gpu-memory-utilization 0.93 --max-model-len 262144 --max-num-seqs 8
```
LMCache's MP connector requires batched tokens in [chunk, 2·chunk). The larger chunk of nvfp4 KV is why its prefill (12.8K t/s at 8K) beats the fp8 tier profile's (9.7K). Let vLLM profile the pool; `--kv-cache-memory` hints ignore warmup transients. The pool varies about ±6% boot to boot on this nightly; the launcher's band is 285–335K.

```bash
--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":4}'
--no-async-scheduling
--mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill
```
MTP `ns=4` with the checkpoint's own draft head. Async scheduling off: about 1 tool-eval point with MTP on this hybrid, and the historical cause of KV corruption under spec decode ([vllm#42655](https://github.com/vllm-project/vllm/issues/42655)). `align` packs the GDN state into the unified KV pool, gives LMCache a scheduler-sized page to store, and is worth about 3 tool-eval points with spec decode (91 with, 87–88.5 without, 2026-08-15 ladder).

```bash
--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
--override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}'
--default-chat-template-kwargs '{"preserve_thinking":true,"reasoning_effort":"medium"}'
--limit-mm-per-prompt '{"image":4,"video":0}'
```
Qwen3.8's template prefills `<think>` and injects a reasoning-effort system line; `qwen3` parses it, `qwen3_xml` is the tool format (`hermes` drops calls). T=0.6 over the model default T=1.0 was measured on 2026-08-14: 90.5 ± 2.1 vs 87.8 ± 1.3 on a 69×4. Reasoning effort `medium` is the daily default; `xhigh` livelocked an eval engine on 2026-08-15. `preserve_thinking` keeps prior `<think>` blocks across turns if the client resends them in the `reasoning` field (not `reasoning_content`, which vLLM ignores on input). The tool parser is on even for harnesses that parse text themselves: a request that carries `tools` without `tool_choice` defaults to `auto` and is rejected without it.

## Host

Memory-only overclock, +4500 MHz VRAM offset at the 600 W limit, core stock; decode is bandwidth-bound so the memory clock is the only knob that moves it. All throughput numbers in this repo are with this offset. Reproduce:

```bash
sudo nvidia-smi -pm 1 && sudo nvidia-smi -pl 600 && sudo nvidia-smi -rgc
sudo python3 -c "import pynvml as N; N.nvmlInit(); h=N.nvmlDeviceGetHandleByIndex(0); N.nvmlDeviceSetGpcClkVfOffset(h,0); N.nvmlDeviceSetMemClkVfOffset(h,4500)"
```

Verify with `nvidia-smi -q -d CLOCK`. Disable swap: over-committing RAM next to a 24 GiB pinned tier hard-hangs the box instead of failing fast.
