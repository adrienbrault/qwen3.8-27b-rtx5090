# NOTES 17 — vLLM v0.29.0rc1 regressions found by the R165 auditions (2026-09-03)

Local record + issue drafts. Nothing here is filed upstream; filing is the maintainer's call.

## 1. DFlash/DSpark on hybrid (Mamba) models: zero prefix-cache hits (upstream, known)
`_annotate_eagle_groups` marks only MLA drafters; unmarked → every KV group is treated as a draft group; NEW in rc1, `MambaManager` honours `drop_eagle_block`, so the widened two-chunk lookup is never satisfied under align mode. Warning at boot: `kv_cache_utils.py:1893`. Fix = PR #54163 (open, Ledgero), ported here as 0134 for rc1 (proof: warm revisit 0.166 s / 53,248 hits vs 7.49 s / 0 without). Nothing to file; #54163 needs a release-branch backport once merged.

## 2. FlashInfer builder: `paged_kv_indices` became a plain tensor (ours, fixed)
rc1 turned `self.paged_kv_indices` into a plain GPU tensor while `paged_kv_indptr` / `paged_kv_last_page_len` stay `CpuGpuBuffer`; the rebased 0101/0109 read `.gpu` on it and crashed in `profile_cudagraph_memory`. Fixed in 0101/0109 (two sites). Not upstream's bug.

## 3. Drafter with an explicit `kv_cache_dtype` never receives the resolved KV layout (upstream, NOT reported)
**Symptom**: `ValueError: KV cache layout has not been resolved yet; it is resolved once by the engine core (resolve_kv_cache_layout) unless explicitly set by the user.` raised from `FlashInferImpl.forward` → `kv_cache_layout` → `CacheConfig.get_resolved_kv_cache_layout`, during `determine_available_memory → profile_cudagraph_memory → speculator.capture()` (MRV2), i.e. at the first drafter forward. Engine core dies at init.

**Repro shape**: any `--speculative-config` with `"kv_cache_dtype"` set (eagle, eagle3, mtp, dflash, dspark) + FlashInfer attention. Ours: Qwen3.8-27B NVFP4 + DFlash2 draft, TP2, `--kv-cache-dtype nvfp4`, spec `{"method":"dflash","model":…,"num_speculative_tokens":9,"draft_tensor_parallel_size":2,"attention_backend":"FLASHINFER","kv_cache_dtype":"nvfp4"}`. Same command without the `kv_cache_dtype` field boots.

**Mechanism (rc1 source)**:
- `vllm/v1/engine/core.py:289-294` resolves the layout before profiling and RPCs `set_kv_cache_layout` → `worker_base.py:112` `record_kv_cache_layout(self.vllm_config.cache_config, …)`; `gpu_worker.py:719` records again on `self.cache_config` from `KVCacheConfig`. Both write the worker's ORIGINAL `CacheConfig`.
- `vllm/v1/worker/gpu/spec_decode/dflash/utils.py:25-39` (and `eagle/utils.py:42-49`, `dspark/utils.py:60-67`, `v1/spec_decode/llm_base_proposer.py:1315-1322`) build the draft model under `replace(vllm_config, cache_config=replace(cache_config, cache_dtype=speculative_config.kv_cache_dtype))` when the field is set. The draft's attention layers capture this COPY (`FlashInferImpl.__init__`: `self.cache_config = vllm_config.cache_config`), whose `kv_cache_layout` stays `None` forever.
- With the field unset the draft shares the target's `CacheConfig` object and inherits the layout.

**Suggested fix (either)**: (a) have `record_kv_cache_layout` reach derived configs — e.g. the drafter loaders register their copy, or the worker records on `speculator.attn_vllm_config.cache_config` too; (b) make `FlashInferImpl.kv_cache_layout` fall back to the `KVCacheConfig.kv_cache_layout` the impl is initialised with rather than the per-model `CacheConfig` copy. (b) is local to the backend and covers all four loaders.

**Draft title**: `[Bug][Spec Decode] v0.29.0rc1: drafters with an explicit speculative kv_cache_dtype crash at profiling — "KV cache layout has not been resolved yet" (cache_config copy never receives the resolved layout)`

## 4. rc1 pool/headroom on the SM120 nvfp4 route (measured, not a bug report)
Graph reserve (#53306/#53955) sizes the pool with the estimate; at util 0.90 the nvfp4 + DFlash2 route has 32 MiB free after boot and dies in the first 8-request 8K prefill (368 MiB + 242 MiB transients); v0.28 at the same util survives (3 MiB reported free, allocator pool covers it). util 0.88 gives 827 MiB headroom and pool 861,565 (vs 984,411 on v0.28 at 0.90). See FINDINGS R165c.
