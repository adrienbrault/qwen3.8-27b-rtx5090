# NOTES19 — rc1 NVFP4 long-context decode regression

## Scope and conclusion

This is a local-only static investigation. I did not run a GPU workload or contact the box. I treat the supplied R167 results—normal short-context performance, normal FP8-KV performance, and an NVFP4-only 30K drop from roughly 144–157 tok/s to 28.95 tok/s—as observations rather than results reproduced here (`PATCHES/patches-v0290/BRIEF19.md:7-18`). In the labels below, **VERIFIED** means that the stated code path or version delta is present in the supplied files; it does not mean that I measured its contribution on RTX 5090 hardware.

The strongest result is a concrete, version-selective code cause: FlashInfer 0.6.18 forcibly disables FA2 split-KV whenever `kv_data_type` is packed NVFP4 (`torch.uint8`), overriding the `disable_split_kv=False` value supplied by vLLM. FlashInfer 0.6.16.post3 honors the caller's `False`. The 0.6.18 scheduler consequently assigns exactly one KV chunk per query tile, so short queries over a 30K KV range lose split-KV's cross-CTA parallelism (`TREES/fi/src-0.6.18/flashinfer/prefill.py:1513-1535`, `TREES/fi/src-0.6.18/flashinfer/prefill.py:2596-2605`, `TREES/fi/src-0.6.16.post3/flashinfer/prefill.py:2472-2477`, `TREES/fi/src-0.6.18/flashinfer/data/include/flashinfer/attention/scheduler.cuh:635-669`). This exactly selects the slow cell's library version, NVFP4 dtype, short-Q/long-KV shape, target speculative verification, and non-causal DFlash attention; it does not select FP8 KV or ordinary single-token decode.

I therefore provide `0136-opt-in-nvfp4-prefill-split-kv-v0290.diff`. It is deliberately opt-in because FlashInfer says its forced gate is an empirical correctness workaround for corrupted outputs and says the underlying bug is unconfirmed (`TREES/fi/src-0.6.18/flashinfer/prefill.py:1513-1535`). The throughput attribution and the safety of re-enabling split-KV still require the one-box A/B and long-context fidelity check described below.

## Ranked root-cause hypotheses

### 1. FlashInfer 0.6.18's NVFP4 prefill guard removes long-KV split parallelism

**Label: VERIFIED (code-level mechanism; hardware attribution pending).**

Evidence:

- On SM120 with an NVFP4 cache, rc1 sets `use_fa2_nvfp4_kv=True` and records the operational cache type as NVFP4 (`TREES/v029real/vllm/v1/attention/backends/flashinfer.py:882-905`). With XQA disabled, the target falls back to native FlashInfer (`TREES/v029real/vllm/v1/attention/backends/flashinfer.py:1002-1006`).
- The vLLM default is `disable_split_kv=False` unless batch-invariant mode is enabled (`TREES/v029real/vllm/v1/attention/backends/flashinfer.py:815-822`). The native prefill plan passes packed NVFP4 as `kv_data_type=torch.uint8` and passes that `False` value (`TREES/v029real/vllm/v1/attention/backends/flashinfer.py:1938-1960`).
- The builder computes uniform speculative verification width as bonus token plus the configured speculative tokens (`TREES/v029real/vllm/v1/attention/backends/flashinfer.py:916-938`), so ns9 has query length ten—greater than the decode threshold of one. `split_decodes_and_prefills` therefore returns zero decodes and sends the whole batch to prefill (`TREES/v029real/vllm/v1/attention/backends/utils.py:798-813`). The prefill CUDA-graph predicate and wrapper selection are in the same build path (`TREES/v029real/vllm/v1/attention/backends/flashinfer.py:1875-1891`).
- FlashInfer 0.6.18 detects `torch.uint8` as NVFP4 and changes the caller's `disable_split_kv=False` to `True` before invoking the FA2 planner (`TREES/fi/src-0.6.18/flashinfer/prefill.py:1513-1535`, `TREES/fi/src-0.6.18/flashinfer/prefill.py:2596-2608`). FlashInfer 0.6.16.post3 has no such override and appends the caller's value directly (`TREES/fi/src-0.6.16.post3/flashinfer/prefill.py:2472-2477`). The equivalent ragged wrapper has the same 0.6.18-only override (`TREES/fi/src-0.6.18/flashinfer/prefill.py:3915-3927`, `TREES/fi/src-0.6.16.post3/flashinfer/prefill.py:3660-3665`).
- In the native scheduler, `disable_split_kv=True` makes the KV chunk effectively unbounded and then hard-codes one chunk; the enabled path instead binary-searches a chunk size and may produce multiple KV tiles (`TREES/fi/src-0.6.18/flashinfer/data/include/flashinfer/attention/scheduler.cuh:635-669`). When partitioning is active the paged prefill kernel writes temporary states and launches `VariableLengthMergeStates`; the unsplit path has no merge (`TREES/fi/src-0.6.18/flashinfer/data/include/flashinfer/attention/prefill.cuh:4483-4509`).
- The gate is dtype-specific: its own code returns true for packed/native FP4 and explicitly says FP8 and 16-bit caches retain split-KV (`TREES/fi/src-0.6.18/flashinfer/prefill.py:1529-1535`). That matches the supplied FP8 control and context scaling (`PATCHES/patches-v0290/BRIEF19.md:7-20`).
- DFlash amplifies the same mechanism rather than escaping it. With no explicit speculative `kv_cache_dtype`, its loader reuses the target `CacheConfig` object (`TREES/v029real/vllm/v1/worker/gpu/spec_decode/dflash/utils.py:20-40`). Its metadata stores the absolute context-plus-query sequence length (`TREES/v029real/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:651-659`), and non-causal attention with dedicated XQA off is classified entirely as prefill (`TREES/v029real/vllm/v1/attention/backends/flashinfer.py:1575-1591`). The DFlash attention layer computes only query-token Q/K/V but reads the already-populated context KV through `Attention` (`TREES/v029real/vllm/model_executor/models/qwen3_dflash.py:243-267`).

**Cheapest confirming measurement:** apply 0136 to rc1 and run the same 30K/c1/ns9 cell twice, first with the default and then with `VLLM_SM12X_NVFP4_PREFILL_SPLIT_KV=1`; retain the existing 32-step torch profiler. Confirmation is a large fall in the approximately 85 ms step time, with `BatchPrefillWithPagedKVCacheKernel` time falling and `VariableLengthMergeStates` appearing in the opt-in profile. Refutation is no material attention-kernel or step-time change. Compare generated tokens/needles and target/draft logits or acceptance as well: speed recovery with any fidelity divergence means this identifies the performance cause but is not a shippable fix.

### 2. rc1 misses the intended FULL/Piecewise CUDA-graph replay for target or drafter

**Label: HYPOTHESIS (ranked well below #1).**

Evidence and evidence against:

- The graph dispatcher source is the same in both applied trees in the inspected decision region: both form uniform FULL keys from padded token count and uniform query length (`TREES/v028p/vllm/v1/cudagraph_dispatcher.py:132-156`, `TREES/v029real/vllm/v1/cudagraph_dispatcher.py:132-156`), and both select FULL, then PIECEWISE, then eager using the same matching rules (`TREES/v028p/vllm/v1/cudagraph_dispatcher.py:235-324`, `TREES/v029real/vllm/v1/cudagraph_dispatcher.py:235-324`).
- The runner's uniform-decode test is also unchanged in substance: it requires the scheduled maximum to equal the configured uniform query length and total tokens to equal that width times requests (`TREES/v028p/vllm/v1/worker/gpu_model_runner.py:3990-4008`, `TREES/v029real/vllm/v1/worker/gpu_model_runner.py:3950-3968`). Both versions pass that result into the dispatcher (`TREES/v028p/vllm/v1/worker/gpu_model_runner.py:4055-4112`, `TREES/v029real/vllm/v1/worker/gpu_model_runner.py:4015-4072`).
- The FlashInfer support calculation is likewise equivalent for this route. With the custom DFlash graph knob enabled, non-causal SM120 NVFP4 reports `UNIFORM_BATCH`; causal SM12x with a supported attention shape also reports `UNIFORM_BATCH` (`TREES/v028p/vllm/v1/attention/backends/flashinfer.py:1133-1204`, `TREES/v029real/vllm/v1/attention/backends/flashinfer.py:1162-1228`). DFlash2 explicitly promotes its attention config for independent FULL-decode capture on this route (`TREES/v029real/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:121-142`).
- The supplied boot log says graph capture completed for this cell, but that proves capture, not runtime key hits (`PATCHES/patches-v0290/BRIEF19.md:43-45`). A runtime descriptor mismatch remains possible, but the static version comparison does not expose one.

**Cheapest confirming measurement:** in the planned 32-step profile, compare per-step `cudaGraphLaunch` counts and the count of individually launched model kernels between v0.28 and rc1. A major rc1 loss of graph launches with many extra individual kernels confirms this branch; equal graph-launch structure while `BatchPrefillWithPagedKVCacheKernel` alone expands toward 85 ms refutes it.

### 3. rc1's page-table staging adds per-step CPU/paging overhead

**Label: HYPOTHESIS.**

Evidence:

- v0.28 keeps `paged_kv_indptr`, indices, and last-page lengths in reusable `CpuGpuBuffer` objects and creates a persistent CPU indptr copy buffer (`TREES/v028p/vllm/v1/attention/backends/flashinfer.py:1059-1069`). Its per-step builder copies through that persistent buffer and runs the GPU page-index gather over the requests (`TREES/v028p/vllm/v1/attention/backends/flashinfer.py:1506-1532`).
- rc1 makes page indices a plain GPU tensor and creates indptr/last-page buffers unpinned (`TREES/v029real/vllm/v1/attention/backends/flashinfer.py:1096-1105`). Each build may call `.pin_memory()` on the small request-count indptr and last-page slices before copying, then runs the page-index gather (`TREES/v029real/vllm/v1/attention/backends/flashinfer.py:1531-1567`).
- This can add CPU allocation/copy overhead, but only the page-index gather scales with the number of pages and its algorithm is present in both versions (`TREES/v028p/vllm/v1/attention/backends/flashinfer.py:1523-1532`, `TREES/v029real/vllm/v1/attention/backends/flashinfer.py:1543-1552`). It is therefore a weaker match than the new split-KV gate.

**Cheapest confirming measurement:** include CPU self-time in the same 32-step profile and sum `aten::pin_memory`, H2D-copy, and `_copy_page_indices_kernel` time per step. This hypothesis needs approximately the missing 70 ms to appear there and grow with page count; sub-millisecond staging refutes it.

### 4. an rc1/0133 GDN decode-kernel change regressed the hybrid layers

**Label: HYPOTHESIS (unlikely to explain context scaling).**

Evidence:

- Both trees select a fused CUDA or Triton GDN decode implementation from the configured kernel and current shape/state support (`TREES/v028p/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:507-529`, `TREES/v029real/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:518-543`).
- rc1 changes packed recurrent beta computation from a BF16 round-trip to FP32 and adds an SM12x launch selector that can use `BV=16` for one narrow head/batch shape; the wrapper launches over batch, key/value heads, and fixed K/V dimensions (`TREES/v028p/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:326-344`, `TREES/v029real/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:331-349`, `TREES/v029real/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:352-380`, `TREES/v029real/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:476-499`). The Qwen call supplies the current token batch and recurrent state, not the historical KV length (`TREES/v029real/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1727-1778`).
- The fused speculative GDN path is gated by speculative metadata, state dtype, head ratio, maximum speculative width, and kernel availability; none of these predicates inspect context length (`TREES/v029real/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1897-1913`). This makes GDN a poor explanation for a 1K/30K divergence.

**Cheapest confirming measurement:** group the profiler's `fused_gdn_decode_post_conv_mtp`, packed recurrent Triton, convolution-update, and GDN norm kernels per step at short and 30K context. Context-proportional growth toward the missing 70 ms confirms; stable counts and durations refute.

### 5. another FlashInfer 0.6.17/0.6.18 FA2 kernel change is slow independently of the wrapper guard

**Label: HYPOTHESIS.**

Evidence:

- The supplied brief identifies FA2 PRs #3890, #4272, and #4389 as part of the library delta (`PATCHES/patches-v0290/BRIEF19.md:23-28`). The available sources confirm #3890's described shape: 0.6.18 adds `chunk_end` and skips `LogitsTransform` beyond the split boundary, whereas 0.6.16.post3 calls the transform for those lanes (`TREES/fi/src-0.6.18/flashinfer/data/include/flashinfer/attention/prefill.cuh:1415-1452`, `TREES/fi/src-0.6.16.post3/flashinfer/data/include/flashinfer/attention/prefill.cuh:1367-1411`).
- That change is not the best primary explanation because the discovered 0.6.18 Python guard prevents split-KV on this exact dtype before kernel launch (`TREES/fi/src-0.6.18/flashinfer/prefill.py:2596-2608`). Other CUDA-side deltas could still matter once the guard is neutralized.

**Cheapest confirming measurement:** after enabling 0136 in both crossed library cells, compare rc1+FI 0.6.18 against rc1+FI 0.6.16.post3. If both now recover, the wrapper guard was sufficient; if 0.6.18 remains materially slower with split-KV demonstrably active, inspect the paged-prefill and merge kernel durations as an independent kernel regression.

### 6. dropping the explicit draft KV dtype changes rc1's drafter to a non-NVFP4 route

**Label: VERIFIED as refuted by the inspected loader path.**

Evidence:

- Supplying an explicit draft dtype makes a copied `CacheConfig`; omitting it reuses `vllm_config.cache_config` itself (`TREES/v029real/vllm/v1/worker/gpu/spec_decode/dflash/utils.py:20-40`). The supplied regression record explains why the explicit field was removed and confirms that the shared-object branch inherits the resolved target layout (`PATCHES/patches-v0290/NOTES17-rc1-regressions.md:11-21`).
- The v0.28 loader has the same dtype-selection branches: its explicit `nvfp4` field creates a config copy whose `cache_dtype` remains NVFP4, whereas rc1 without the field shares the target NVFP4 config (`TREES/v028p/vllm/v1/worker/gpu/spec_decode/dflash/utils.py:31-45`, `TREES/v029real/vllm/v1/worker/gpu/spec_decode/dflash/utils.py:25-40`). The launch change affects config object identity and layout resolution, not the requested dtype.
- The DFlash layers receive that current draft vLLM config's cache config (`TREES/v029real/vllm/model_executor/models/qwen3_dflash.py:430-450`). On the stated launch, the target config is NVFP4 and the rc1 boot reports `kv_cache_dtype=nvfp4` (`PATCHES/patches-v0290/BRIEF19.md:40-45`). Thus removing the field avoids the layout-copy bug; it does not select FP8 or automatic KV storage in the code inspected.

**Cheapest confirming measurement:** print the target and draft builders' `cache_config.cache_dtype`, resolved layout, `is_kvcache_nvfp4`, and `use_fa2_nvfp4_kv` once at boot. Any non-NVFP4 draft value revives this hypothesis; matching NVFP4/FA2 values refute it operationally.

## Required v0.28 versus rc1 path comparison

### Native FA2 attention and planning

- Wrapper construction is materially the same for native decode: both choose backend `fa2` for SM120 NVFP4, set `use_tensor_cores=True`, and bind persistent paged metadata buffers for CUDA graphs (`TREES/v028p/vllm/v1/attention/backends/flashinfer.py:1439-1478`, `TREES/v029real/vllm/v1/attention/backends/flashinfer.py:1463-1501`). Both pass Q dtype, packed-uint8 KV dtype, output dtype, fixed split, and `disable_split_kv` into their planner (`TREES/v028p/vllm/v1/attention/backends/flashinfer.py:2000-2040`, `TREES/v029real/vllm/v1/attention/backends/flashinfer.py:2030-2070`).
- Their local `fast_plan_decode` implementations are identical in behavior: dynamic mode invokes public `plan()` every call; graph mode invokes it only for the first call and subsequently uses FlashInfer's fast planner (`TREES/v028p/vllm/v1/attention/backends/flashinfer.py:3042-3096`, `TREES/v029real/vllm/v1/attention/backends/flashinfer.py:3094-3140`). Thus rc1 did not newly start public decode planning every graph replay.
- Native speculative verification is a prefill, not decode, and both versions call `prefill_wrapper.plan()` in every metadata build (`TREES/v028p/vllm/v1/attention/backends/flashinfer.py:1851-1938`, `TREES/v029real/vllm/v1/attention/backends/flashinfer.py:1875-1961`). That per-step behavior is not an rc1 vLLM delta; the consequential delta is what FlashInfer 0.6.18 does to its prefill `disable_split_kv` argument.
- 0131 is present in both applied trees. It only shrinks each pooled wrapper's private integer workspace before any plan; both retain the same separately sized shared float workspace (`TREES/v028p/vllm/v1/attention/backends/flashinfer.py:124-155`, `TREES/v029real/vllm/v1/attention/backends/flashinfer.py:161-192`, `TREES/v028p/vllm/v1/attention/backends/flashinfer.py:1206-1229`, `TREES/v029real/vllm/v1/attention/backends/flashinfer.py:1230-1253`). It does not introduce rc1-only planning or an O(context) buffer resize.

### FlashInfer API compatibility and pinning

- The relevant prefill wrapper constructor retains the same `kv_layout`, graph-buffer, backend, and JIT arguments in 0.6.16.post3 and 0.6.18 (`TREES/fi/src-0.6.16.post3/flashinfer/prefill.py:1600-1610`, `TREES/fi/src-0.6.18/flashinfer/prefill.py:1645-1655`). The relevant plan arguments—including `use_fp16_qk_reduction`, `q_data_type`, `kv_data_type`, `o_data_type`, `fixed_split_size`, and `disable_split_kv`—are also present in both (`TREES/fi/src-0.6.16.post3/flashinfer/prefill.py:2078-2105`, `TREES/fi/src-0.6.18/flashinfer/prefill.py:2125-2156`). Their defaults did not create the regression because vLLM passes the dtype and split flag explicitly.
- The rc1 Qwen attention backend's top-level wrapper imports are all exported by 0.6.16.post3 (`TREES/v029real/vllm/v1/attention/backends/flashinfer.py:14-23`, `TREES/fi/src-0.6.16.post3/flashinfer/__init__.py:31-58`, `TREES/fi/src-0.6.16.post3/flashinfer/__init__.py:165-180`). The 0.6.16 fast decode planner accepts the kwargs used here, including both dtypes, fixed split, and the split-disable flag (`TREES/fi/src-0.6.16.post3/flashinfer/decode.py:3729-3752`). Qwen GDN's imported `chunk_gated_delta_rule` also exists in 0.6.16.post3 (`TREES/v029real/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:181-220`, `TREES/fi/src-0.6.16.post3/flashinfer/gdn_prefill.py:103-119`). I found no missing 0.6.16 API in this workload's inspected path.
- A general rc1 downgrade is nevertheless unsupported: rc1 pins both `flashinfer-python` and `flashinfer-cubin` exactly to 0.6.18 (`TREES/v029real/requirements/cuda.txt:13-18`). If the crossed-arm test uses 0.6.16.post3, replace Python and cubin as a matched pair, clear only that environment's FI JIT cache, and treat it as a diagnostic/temporary deployment pin rather than a package-wide compatibility claim. New rc1 models, communication backends, and dynamically loaded optional paths were outside this workload-path proof.

No supplied file attributes the new `_nvfp4_kv_requires_disabled_split_kv` wrapper gate to a PR. The concrete cause identified here is that Python wrapper gate, not one of the named CUDA PRs. Its own comment calls the underlying NVFP4 split corruption unconfirmed (`TREES/fi/src-0.6.18/flashinfer/prefill.py:1513-1529`), so assigning that deeper bug to #3890, #4272, or #4389 would be speculation.

## Reading the planned r168 matrix

| Cell/check | Result that supports #1 | Result that points elsewhere |
|---|---|---|
| Crossed FI versions | v0.28+0.6.18 becomes slow and rc1+0.6.16 becomes fast. | Performance follows the vLLM tree rather than FI. |
| Spec OFF at 30K | Both versions are near parity: q=1 takes native decode, avoiding the new prefill gate (`TREES/v029real/vllm/v1/attention/backends/utils.py:803-808`, `TREES/v029real/vllm/v1/attention/backends/flashinfer.py:2020-2070`). | rc1 remains slow: inspect native `BatchDecodeWithPagedKVCacheWrapper`/`fast_decode_plan`, not only prefill. |
| Masked XQA on rc1 | Partial or full recovery means target verification contributes; the non-causal drafter can still remain on prefill (`TREES/v029real/vllm/v1/attention/backends/flashinfer.py:1577-1591`, `TREES/v029real/vllm/v1/attention/backends/flashinfer.py:1990-2019`). | No recovery means the drafter or another component dominates, subject to confirming XQA actually selected. |
| FP8 control | It stays fast because the 0.6.18 force-disable helper excludes FP8 (`TREES/fi/src-0.6.18/flashinfer/prefill.py:1529-1535`). | FP8 also slows under the same graph/path: investigate graph dispatch, planning, or a dtype-independent kernel delta. |
| Profiler | rc1 NVFP4 time is concentrated in `BatchPrefillWithPagedKVCacheKernel`; 0.6.16 or 0136 adds `VariableLengthMergeStates` and sharply shortens the paged-prefill kernel (`TREES/fi/src-0.6.18/flashinfer/data/include/flashinfer/attention/prefill.cuh:4453-4509`). | The missing time is instead CPU page staging, GDN, collectives, many eager launches, or a native decode kernel. |

## Patch and operational recommendation

`0136-opt-in-nvfp4-prefill-split-kv-v0290.diff` adds `VLLM_SM12X_NVFP4_PREFILL_SPLIT_KV=1`. On the SM120 NVFP4 route it neutralizes only FlashInfer 0.6.18's private force-disable helper, restoring the 0.6.16 caller-controlled behavior. With the variable unset, behavior is unchanged. `verify-0136.sh` checks a clean `patch -p1 --fuzz=0` application, applies it to a temporary copy, compiles the patched Python, and executes the helper against a fake FlashInfer module to verify both safe-default and opt-in behavior; it performs no network or GPU operation.

Recommended order on hardware: first run the crossed FI matrix and the 0136 env-off/env-on A/B; then require long-context needle/logit equivalence before considering the knob deployable. If split-KV restores speed but fails fidelity, keep the knob off and either pin the matched 0.6.16.post3 Python+cubin pair for this audited workload or pursue a FlashInfer kernel fix for the corruption that motivated the 0.6.18 guard. Merely passing `disable_split_kv=False` or choosing a fixed split size cannot bypass 0.6.18, because the wrapper changes the flag to `True` after receiving both values (`TREES/fi/src-0.6.18/flashinfer/prefill.py:2596-2605`).
