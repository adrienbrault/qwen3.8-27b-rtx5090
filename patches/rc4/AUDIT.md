# vLLM 0.27 / LMCache 0.5.3 patch audit

## Scope and verdicts

This audit is based only on the pristine trees shipped in this directory:

- `lmcache-0.5.3/lmcache-0.5.3/`
- `vllm-nightly/vllm/`

No network, container, GPU, or remote-host operation was used.

| Patch | Verdict | Result |
|---|---|---|
| 0001 | **re-derived; needs runtime verification** | LMCache 0.5.3 still only edits rank-5 split-K/V subpages. The rebase adds rank-4 fused-K/V support, but vLLM 0.27's registration/normalization contract changed and the supplied vLLM snapshot omits the backend allocation code needed to prove the production stride. |
| 0002 | **re-derived; needs runtime verification** | LMCache 0.5.3 still rejects the non-contiguous dense-permutation view. The rebase uses a checked `as_strided` regroup and updates the mutable layout hint to the normalized physical NHD layout. |
| 0003 | **obsolete / fixed upstream** | vLLM 0.27 uses a fixed-point common hybrid hit and defaults connectors to non-divergent hits. LMCache does not opt into divergent local hybrid hits. No port produced. |
| 0005 | **kept-rebased** | The connector-blind Eagle subtraction is still present. Rebased to the expanded 0.27 split routine. |
| 0007 | **kept-rebased** | The staging batch is still hard-coded to four. Rebased for the moved base cache-context import. |
| 0008 | **kept-rebased** | `max_capacity_gb` is still not passed to native code and native stores still have no reservation/accounting. Rebased with the 0.5.3 ObjectKey wire format accounted for. |
| 0009 | **new; kept** | Adds native byte-accurate watermark LRU, orphan-temp indexing/eviction, reserve-before-write admission, lookup/read pins, and exact native usage reporting. |
| 0010 | **new; kept** | Both abort-racy scheduler assertions are still present; converted to warning plus `continue`. |

## The decisive fp8 layout answer

**The vLLM-0.27 side should not be treated as mechanically identical to the old 0001/0002 target.** The scheduler still plans a logical hybrid page, and the general multi-group allocator still uses a uniform shared page size, but the connector-facing attention contract is now a standardized rank-4, blocks-first, fused-content tensor.

Pristine vLLM evidence:

`vllm-nightly/vllm/distributed/kv_transfer/kv_connector/utils.py:430-451`

```python
# Figure out whether the first dimension of the cache is K/V
# or num_blocks.
...
assert kv_cache_shape[0] == 1, (
    "KV cache layout must be blocks-first; expected mocked "
...
assert len(kv_cache_shape) == 4, (
    "Attention KV cache layout must be standardized as "
    "[num_blocks, num_kv_heads, block_size, content_size], "
```

`vllm-nightly/vllm/distributed/kv_transfer/kv_connector/utils.py:514-522`

```python
# Whether to logically split each block into two separately-indexable
# sub-regions. With K and V packed into the content dim, an attention
# block transfers as a single unit — no K/V sub-split is needed.
...
return self.is_mamba and not self._cross_layers_blocks
```

For any configuration that is neither all-`UniformTypeKVCacheSpecs` nor explicitly opted into cross-layer packing, vLLM still selects the general shared-pool allocator:

`vllm-nightly/vllm/v1/core/kv_cache_utils.py:1274-1293`

```python
def _use_packed_kv_cache_config(...):
    is_dsv4 = all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
...
    return is_dsv4 or (enable_cross_layers and len(kv_cache_groups) > 1)
```

`vllm-nightly/vllm/v1/core/kv_cache_utils.py:1377-1403`

```python
else:
    # General case:
    # We will have group_size memory pools, each is shared by one layer from
    # each group.
...
    page_size = get_uniform_page_size(
        [group.kv_cache_spec for group in kv_cache_groups]
    )
...
    KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
```

LMCache 0.5.3 also changed what happens after registration. It now carries a vLLM layout hint through the edit and format-discovery paths:

`lmcache-0.5.3/lmcache-0.5.3/lmcache/integration/vllm/lmcache_mp_connector.py:802-816`

```python
layout_hints = vllm_layout_hints()
kv_caches = apply_kv_cache_group_edits(
    kv_cache_config, kv_caches, layout_hints=layout_hints
)
engine_group_infos = create_engine_group_infos_from_vllm(
    kv_cache_config,
    kv_caches,
    layout_hints=layout_hints,
)
self.worker_adapter.register_kv_caches(
    kv_caches, engine_group_infos=engine_group_infos
)
```

It normalizes exposed permutations by stride before format detection:

`lmcache-0.5.3/lmcache-0.5.3/lmcache/v1/gpu_connector/kv_format/contiguity.py:43-80`

```python
def attempt_permute_to_contiguous_view(...):
    """Return a contiguous view ... metadata-only (no copy).
...
    strides = kv_caches.stride()
    perm = sorted(range(kv_caches.ndim), key=lambda i: strides[i], reverse=True)
    result = kv_caches.permute(perm)
    if result.is_contiguous():
        return result.view(_get_expected_shape(result.stride(), result.numel()))
```

Its vLLM detector then interprets any rank-4 attention cache according to `kv_layout`:

`lmcache-0.5.3/lmcache-0.5.3/lmcache/v1/gpu_connector/kv_format/detectors/vllm.py:37-51`

```python
# Blocks-first fused K/V is the only rank-4 vLLM layout...
# The two middle axes are NH/BS (HND) or BS/NH (NHD)
...
if kv_caches[0].dim() == 4:
    if is_hnd:
        return lmc_ops.EngineKVFormat.NL_X_NB_NH_BS_CS, kv_caches
    return lmc_ops.EngineKVFormat.NL_X_NB_BS_NH_CS, kv_caches
```

The supplied vLLM snapshot does **not** include `vllm/v1/worker/utils.py`, the selected attention backend's allocator, or the worker block-table expansion code. Therefore it cannot prove statically that this nightly still exposes the production Qwen tensor as exactly 16-token kernel pages inside each 1616-token logical page, nor that its real stride remains `[32768, 512, 2048, 1]`. This is why 0001/0002 are re-derived, not called mechanical rebases.

### Required runtime gate for 0001/0002

Before production, instrument `LMCacheConnectorV1Worker.register_kv_caches` immediately before and after `apply_kv_cache_group_edits` and log, for every full-attention layer:

- layer name, `shape`, `stride`, `storage_offset`, `dtype`;
- `spec.block_size` and `spec.page_size_bytes`;
- `layout_hints` before and after the edit;
- the final detected `EngineKVFormat`.

For the expected rig layout, confirm that the pre-edit rank-4 tensor has one unique middle axis of 16, that `shape[0]` is divisible by 101, and that one kernel-page byte count times 101 equals the 1616-token spec page. Confirm the edited/normalized tensor has logical block size 1616, aliases the same storage, and is detected as physical NHD after the stride permutation.

Then run the available needle-retrieval + killer-burst gauntlet as a true external-cache test: cold store, process restart, retrieve, and compare all needles; include concurrency/burst pressure. A coherent decode or a positive hit count is not sufficient.

## Patch-by-patch audit

### 0001 — fused rank-4 hybrid subpage view

**Verdict: still needed, re-derived, runtime verification required.**

LMCache 0.5.3 does contain the upstream subpage edit, but it is still rank-5 only:

`lmcache-0.5.3/lmcache-0.5.3/lmcache/integration/vllm/kv_cache_group_edits.py:293-304`

```python
def matches(...):
    return (
...
        and isinstance(kv_cache, torch.Tensor)
        and kv_cache.ndim == 5
        and kv_cache.shape[2] != spec.block_size
    )
```

The apply path likewise assumes split K/V:

`lmcache-0.5.3/lmcache-0.5.3/lmcache/integration/vllm/kv_cache_group_edits.py:326-337`

```python
if not isinstance(kv_cache, torch.Tensor) or kv_cache.shape[1] != 2:
...
kernel_block_size = kv_cache.shape[2]
```

Thus the rank-4 fused cache recognized by the new detector is not regrouped before group metadata is derived. The rebased patch:

- accepts rank-4 fused-content tensors;
- identifies the unique kernel-block middle axis by exact logical-page byte accounting;
- keeps the rank-5 path;
- uses the new three-argument edit signature with `layout_hints`;
- uses 0.5.3's `*_CS` engine-format model rather than the removed `*_TWO_HS` test API.

Output: `rebased/0001-fix-fused-hybrid-subpage-view.diff`.

### 0002 — non-contiguous fp8 regroup

**Verdict: still needed, re-derived, runtime verification required.**

The pristine 0.5.3 edit still rejects every non-contiguous subpaged tensor:

`lmcache-0.5.3/lmcache-0.5.3/lmcache/integration/vllm/kv_cache_group_edits.py:351-360`

```python
kernel_page_bytes = kv_cache.shape[1:].numel() * kv_cache.element_size()
...
if not kv_cache.is_contiguous():
    raise ValueError(
        "kernel-paged attention KV tensor must be contiguous to "
        "re-view as logical pages"
    )
```

The re-derived patch validates that each kernel page is a dense inner-axis permutation, that pages are adjacent, and that the kernel-token axis is the outer physical inner-page axis before creating an `as_strided` logical-page view.

0.5.3-specific change: after that regroup, `attempt_permute_to_contiguous_view` converts the exposed HND-shaped view to physical NHD axes. 0002 therefore also changes the shared mutable `layout_hints["kv_layout"]` to `"NHD"` so the new rank-4 detector cannot reinterpret the normalized axes as HND.

Output: `rebased/0002-strided-fp8-regroup.diff`.

### 0003 — connector Eagle hybrid hit reduction

**Verdict: obsolete; fixed upstream. No rebased patch.**

The old connector-only `max(per_group_hits)` path is gone. vLLM 0.27's common coordinator explicitly uses a monotone fixed point:

`vllm-nightly/vllm/v1/core/kv_cache_coordinator.py:751-762`

```python
def find_longest_cache_hit(...):
    """
    Find the longest cache hit using an iterative fixed-point algorithm.

    Each attention type either accepts the current candidate length or
    reduces it. If any type reduces the length, restart checks over all
    types. This converges because length monotonically decreases...
```

Its Eagle margin specifically excludes Mamba:

`vllm-nightly/vllm/v1/core/kv_cache_coordinator.py:813-845`

```python
drop_eagle_block = use_eagle and idx not in eagle_verified
...
# No margin for
# mamba: its finder never drops ...
if drop_eagle_block and not isinstance(spec, MambaSpec):
...
hit_blocks, _new_hit_length = manager_cls.find_longest_cache_hit(
...
    drop_eagle_block=drop_eagle_block,
)
```

Connector divergence is now an explicit opt-in whose base default is false:

`vllm-nightly/vllm/distributed/kv_transfer/kv_connector/v1/base.py:176-183`

```python
@property
def supports_divergent_local_hybrid_hits(self) -> bool:
    """Whether external hits can complete divergent local hybrid hits.
...
    """
    return False
```

The scheduler uses the safe common result unless a connector opts in:

`vllm-nightly/vllm/v1/core/sched/scheduler.py:443-453`

```python
connector = self.connector
if connector is not None and connector.supports_divergent_local_hybrid_hits:
    return self.kv_cache_manager.get_computed_blocks_for_connector(request)

blocks, num_local, shared_prefix_boundary = (
    self.kv_cache_manager.get_computed_blocks(request)
)
return blocks, num_local, shared_prefix_boundary, False
```

The supplied LMCache connector does not override this property. Therefore its connector path uses the common fixed-point hit, and porting 0003 would duplicate/undermine the new upstream design.

### 0005 — residual Mamba connector prefill boundary

**Verdict: still needed; kept-rebased.**

The exact connector-blind subtraction remains:

`vllm-nightly/vllm/v1/core/sched/scheduler.py:392-400`

```python
last_cache_position = request.num_tokens - request.num_tokens % block_size
if self.use_eagle:
    last_cache_position = max(last_cache_position - block_size, 0)

end = start + num_new_tokens
```

For block size 16, a connector prefill starting at token 16 with a 45-token prompt computes a real final cache boundary of 32. Eagle reduces it to 16, so no mandatory stop lies strictly inside the 16→45 final step; the step crosses token 32 and can export the null Mamba slot.

LMCache 0.5.3 added a launch-time alignment validation:

`lmcache-0.5.3/lmcache-0.5.3/lmcache/integration/vllm/lmcache_mp_connector.py:145-178`

```python
Requiring `block_size <= max_num_batched_tokens < 2 * block_size` makes
vLLM's block-aligned splitting ...
...
if not (block_size <= max_batched < 2 * block_size):
    raise ValueError(...)
```

and invokes it at connector construction (`lmcache_mp_connector.py:600-604`), but that bound does not change the final-step Eagle subtraction shown above.

The target method now accepts local/external computed-token offsets and includes partial-tail/shared-prefix stops. The rebase preserves all of that and changes only the Eagle condition to `self.use_eagle and self.connector is None`, with its stdlib AST regression updated for the new method fields.

Output: `rebased/0005-vllm-residual-mamba-connector-prefill-boundary.diff`.

### 0007 — configurable sidecar GPU staging batch

**Verdict: still needed; kept-rebased.**

The allocation is still hard-coded:

`lmcache-0.5.3/lmcache-0.5.3/lmcache/v1/platform/cuda/cache_context.py:405-411`

```python
self._temp_buffer = _TempGPUBuffer(
    kv_layer_groups_manager=self.kv_layer_groups_manager_,
    lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
    device=self.device_,
    max_batch_size=4,
)
```

The target import moved from `lmcache.v1.platform.base_cache_context` to `lmcache.v1.platform.base.cache_context` (`cache_context.py:39`), which is the only source-anchor change required by the rebase. The port adds the validated `LMCACHE_MP_GPU_STAGING_BATCH_SIZE` helper and CPU tests.

Output: `rebased/0007-sidecar-vram-staging-batch.diff`.

### 0008 — fs_native hard-cap admission

**Verdict: still needed; kept-rebased.**

The native connector has no capacity fields at all:

`lmcache-0.5.3/lmcache-0.5.3/csrc/storage_backends/fs/connector.h:35-49,72-77`

```cpp
FSConnector(..., size_t read_ahead_size = 0);
...
std::string base_path_;
...
size_t read_ahead_size_;
```

Its store path only checks existence, writes, and renames; it does not reserve bytes:

`lmcache-0.5.3/lmcache-0.5.3/csrc/storage_backends/fs/connector.cpp:223-241`

```cpp
// Skip if already stored on disk
if (std::filesystem::exists(file_path)) {
  return;
}
// Determine temp file path
```

The Python config parses `max_capacity_gb`, but the factory passes only five native arguments:

`lmcache-0.5.3/lmcache-0.5.3/lmcache/v1/distributed/l2_adapters/fs_native_l2_adapter.py:146-153`

```python
native_client = LMCacheFSClient(
    config.base_path,
    config.num_workers,
    config.relative_tmp_dir,
    config.use_odirect,
    config.read_ahead_size or 0,
)
```

Target-specific rebase issue: 0.5.3's Python serializer now emits four unsalted or five salted fields:

`lmcache-0.5.3/lmcache-0.5.3/lmcache/v1/distributed/l2_adapters/native_connector_l2_adapter.py:51-68`

```python
<model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>
...
<model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>@<cache_salt>
```

but the pristine C++ header still documents the old three/four-field wire format (`csrc/storage_backends/fs/connector.h:54-60`). The rebase updates the native parser to 0.5.3's object-group-aware shape; otherwise actual adapter-generated keys would be malformed or misparsed.

0008 remains the hard-cap layer: startup scan, atomic reservation before write, rollback, duplicate-store suppression, and delete accounting. 0009 builds on these invariants.

Output: `rebased/0008-fs-native-cap-enforcement.diff`.

### 0009 — fs_native watermark LRU

**Verdict: new; implemented.**

The config keys already exist exactly as requested:

`lmcache-0.5.3/lmcache-0.5.3/lmcache/v1/distributed/l2_adapters/config.py:230-267`

```python
"eviction": {
    "eviction_policy": "LRU",
    "trigger_watermark": 0.8,
    "eviction_ratio": 0.2
}
...
return EvictionConfig(
    eviction_policy=policy,
    trigger_watermark=float(...),
    eviction_ratio=float(...),
)
```

The existing shared controller triggers on usage but asks the policy to evict a ratio:

`lmcache-0.5.3/lmcache-0.5.3/lmcache/v1/distributed/storage_controllers/eviction_controller.py:319-345`

```python
current_usage = state.adapter.get_usage().usage_fraction
if current_usage < 0 or current_usage < watermark:
    return
...
actions = state.eviction_policy.get_eviction_actions(eviction_ratio)
```

and the LRU policy interprets that as a ratio of Python-tracked key count, not bytes-to-target:

`lmcache-0.5.3/lmcache-0.5.3/lmcache/v1/distributed/eviction_policy/lru.py:164-188`

```python
expected_ratio = max(0.0, min(1.0, expected_ratio))
target_count = int(len(self._order) * expected_ratio)
...
for key in self._order:
...
    if len(keys_to_evict) >= target_count:
        break
```

That cannot see restart files, orphaned temps, or successful per-file writes from a batch whose overall completion failed. The new patch therefore puts global LRU beside the authoritative native byte reservation:

- scans and indexes connector-owned `.data` and `.tmp` files at startup, oldest mtime first;
- triggers on **projected** usage at or above `trigger_watermark`;
- evicts least-recently-used files until the new reservation fits at `trigger_watermark - eviction_ratio`;
- records every successful native file independently, so partial-batch files remain evictable even if Python sees a failed batch;
- counts and evicts orphan temp files;
- protects byte count, LRU records, pending writes, and pin counts with one mutex;
- pins successful lookup-and-lock results across the lookup→load interval, adds an active read pin, and releases native lookup pins from Python `submit_unlock`;
- keeps atomic temp-write/rename and 0008 reserve-before-write rollback;
- disables the duplicate Python global controller only for native-managed LRU; `IsolatedLRU` and `noop` keep their existing paths;
- reports exact native global usage.

A single object larger than the target fraction but no larger than the hard cap is admitted after all older evictable files are removed; it cannot mathematically fit below the target without rejecting the object. Normal KV chunks are far below this scale.

Output: `0009-fs-native-watermark-eviction.diff`.

### 0010 — KV transfer completion after abort

**Verdict: new; implemented.**

Both assertions are present in the supplied scheduler:

`vllm-nightly/vllm/v1/core/sched/scheduler.py:2816-2829`

```python
for req_id in kv_connector_output.finished_recving or ():
...
    assert req_id in self.requests
...
for req_id in kv_connector_output.finished_sending or ():
...
    assert req_id in self.requests
```

An asynchronously completed transfer may legitimately outlive an aborted/removed request. The patch converts each assertion to a warning and `continue`; existing requests retain the original receive/send handling.

Output: `rebased/0010-scheduler-xfer-abort.diff`.

## Verification

Clean-apply verification was performed against fresh copies of the pristine supplied trees, in this dependency order:

| Target | Patch | `git apply --check` |
|---|---|---|
| LMCache 0.5.3 | rebased/0001-fix-fused-hybrid-subpage-view.diff | PASS |
| LMCache 0.5.3 + 0001 | rebased/0002-strided-fp8-regroup.diff | PASS |
| LMCache 0.5.3 + prior LM patches | rebased/0007-sidecar-vram-staging-batch.diff | PASS |
| LMCache 0.5.3 + prior LM patches | rebased/0008-fs-native-cap-enforcement.diff | PASS |
| LMCache 0.5.3 + 0008 | 0009-fs-native-watermark-eviction.diff | PASS |
| supplied vLLM nightly | rebased/0005-vllm-residual-mamba-connector-prefill-boundary.diff | PASS |
| supplied vLLM nightly + 0005 | rebased/0010-scheduler-xfer-abort.diff | PASS |

Additional checks:

- Python `py_compile` on all changed runtime modules: PASS.
- `clang++ -std=c++17 -Wall -Wextra -Werror -fsyntax-only` on the final fs connector: PASS.
- 0005's standalone CPU regression: PASS (`PASS: connector + MTP preserves each storable Mamba boundary`).
- Standalone compiled C++ 0009 smoke: PASS for 0.8 trigger, 0.6 post-reservation target, LRU recency, native lookup pins, and exact 60-byte usage.
- Standalone compiled C++ orphan smoke: PASS for restart accounting and eviction of a 50-byte orphan temp before a 40-byte store.
- The LMCache pytest suite was not run because the host Python has neither `pytest` nor `torch`. No dependency was installed, consistent with the no-network/no-environment-mutation constraint.
- No container build or GPU runtime was attempted, as required.

## Dockerfile

`DOCKERFILE.new`:

- starts from the exact requested vLLM image digest;
- extracts the supplied LMCache 0.5.3 sdist;
- applies LMCache patches in dependency order 0001, 0002, 0007, 0008, 0009;
- rebuilds/install LMCache from source with the old Dockerfile's CUDA include/library discovery, `TORCH_CUDA_ARCH_LIST="12.0+PTX"`, `MAX_JOBS=4`, `--no-deps`, and `--no-build-isolation`, so the 0008/0009 C++ extension changes are compiled;
- applies only the still-needed vLLM patches 0005 and 0010 to the installed vLLM package.
