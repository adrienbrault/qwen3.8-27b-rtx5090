# LMCache tier patches — what makes DRAM/NVMe KV offload correct on this hybrid

These six patches are what stand between "LMCache appears to work" and "LMCache is actually
faithful" on a **fp8-KV hybrid model with MTP**. They are **local, pre-upstream**. Build them
with [`Dockerfile`](Dockerfile) here, on top of [`../Dockerfile`](../Dockerfile).

The through-line: every one of these bugs is *silent*. Hit counters go up, output stays fluent,
no errors are logged — and distant facts are gone, or turns come back empty, or your root
filesystem fills up. **Needle-test across a restart before trusting any external KV tier on a
hybrid model.** Coherence and hit rates will lie to you; they lied to us for four rounds.

| patch | project | what it fixes | how it fails without it |
|---|---|---|---|
| `0001-fix-fused-hybrid-subpage-view.diff` | LMCache | Fused rank-4 hybrid page → sub-page view. vLLM's fp8 attention backend registers each hybrid-aligned 1616-token page as ~100 contiguous **16-token kernel pages**; LMCache's regrouping only matched the rank-5 split-K/V layout. | Misreads the 16-vs-1616 slots ratio as *compression* and transfers **one 16-token page per logical block**, wrongly addressed, zero errors logged. Fluent output, vanished needles. |
| `0002-strided-fp8-regroup.diff` | LMCache | The stride-aware regroup itself for that layout. | Same as above — 0001 is the view, 0002 is the copy. |
| `0003-vllm-connector-eagle-hybrid-hit.diff` | **vLLM** | Connector-path prefix-hit reduction. With MTP, `SpeculativeConfig.use_eagle()` marks *all* cache groups EAGLE; on the connector lookup path `FullAttentionManager` honors `drop_eagle_block` (hit `H−B`) while `MambaManager` ignores it (hit `H`), and the scheduler takes `max()` instead of the fixed-point reduction the native path uses. | One attention block `[H−B, H)` is **allocated but never filled** — APC skips it, LMCache's overlap protection skips it, nothing recomputes it. Every request with a local prefix hit decodes over a garbage KV block. **≈10 tool-eval points**, in either connector role, reproducible at concurrency 1. Phenotype is truncated/empty turns and malformed tool calls — *not* incoherence. |
| `0005-vllm-residual-mamba-connector-prefill-boundary.diff` | **vLLM** | Connector-active prefills stop at storable Mamba boundaries. On the final MTP prefill step, EAGLE's one-block subtraction lets a step cross a Mamba block boundary; vLLM inserts its **null Mamba block (ID 0)** in the skipped slot (native APC refuses to hash null entries) but LMCache's store planner exports the block table blindly. | **Null-block bytes stored under a valid token hash.** Later retrieves restore garbage recurrent state at that boundary → instant-EOS / truncation. Caught live by a 3-turn record/replay repro. 0003 alone made it *worse* (a correct local hit extended via a poisoned external chunk). |
| `0007-sidecar-vram-staging-batch.diff` | LMCache | Makes the sidecar's GPU staging batch configurable (`LMCACHE_MP_GPU_STAGING_BATCH_SIZE`). | Sidecar pins **1,412 MiB** of VRAM that `--gpu-memory-utilization` cannot see. With `=1` + `CUDA_MODULE_LOADING=LAZY`: **796 MiB**, zero latency cost — worth ~25K tokens of KV pool. |
| `0008-fs-native-cap-enforcement.diff` | LMCache | Reserve-before-write admission control in the `fs_native` L2 adapter (`csrc` — this one is a real recompile). | `max_capacity_gb` is **telemetry only** and per-adapter eviction is opt-in. Measured: L2 grew to **876 GB against a configured 60 GB cap** and filled the host root filesystem. An unenforced cap on your root disk is a self-brick timer. |

A seventh patch, `0006-tq-profiler-cache-dtype.diff`, is **not in this image** — it propagates the
configured `kv_cache_dtype` into vLLM's CUDA-graph memory profiler, which only matters for
TurboQuant KV (it unblocked TQ's boot; TQ then failed on merit and is closed — see
[../../docs/HISTORY.md](../../docs/HISTORY.md)). It lives in the debug workspace, not here.

## Known limitations of 0008 (honest edges)

External review surfaced two accounting gaps in 0008's admission control, both real, neither yet fixed:

1. **Partial-batch admission can desynchronize disk contents from eviction accounting.** Admission is per-file across concurrent workers, but the Python adapter only records stored keys when the *whole* batch completes successfully. A batch that writes some files and then hits the cap leaves those files consuming capacity while absent from `_key_sizes` and the LRU index — undiscoverable, unevictable, and (in the worst case) capable of pinning the cache at "full, nothing to evict". The proper fix is atomic whole-batch reservation or per-key SET results with per-key accounting; not yet implemented.
2. **Abandoned temp files are counted against capacity at restart but never indexed or cleaned** — a crash mid-write permanently shrinks usable capacity until someone deletes them.

Mitigations until fixed: `serve.sh` sweeps stale temp files from the L2 dir before every boot, and the first-day `du -sh` watch below applies doubly. In ~900 soak cycles plus the 500-task SWE campaign the desync case was not observed to wedge the cache — but the window is real, and this note exists so you don't discover it in production.

## Also required at launch (not patchable)

- `"eviction": {"eviction_policy": "LRU", "trigger_watermark": 0.8, "eviction_ratio": 0.2}` in the
  `fs_native` L2 adapter JSON. 0008 enforces the cap; this block is what actually evicts. Both.
- **Wipe any L2 namespace written by an unpatched build.** Poisoned chunks are not repaired —
  0005 stops new ones being written, it does not heal old ones.

## Upstream status

Not filed. Reports are drafted locally for the vLLM side (0003, 0005) and the LMCache side
(0001, 0002, 0007, 0008), pending a decision to file. Until then these are local patches, and
the version-sensitivity warning in [../README.md](../README.md) applies double: a vLLM nightly
that moves `scheduler.py` or `kv_cache_coordinator.py` will fail the build gate loudly (good)
or shift the anchor (bad). Re-run the in-image regressions after every bump.
