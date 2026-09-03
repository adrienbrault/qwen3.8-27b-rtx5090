# BRIEF20 — capacity-bounded LRU eviction for vLLM's fs (disk) secondary tier

vLLM's OffloadingConnector + TieringOffloadingSpec (`vllm/v1/kv_offload/tiering/`) has a CPU primary tier with an
eviction policy and secondary tiers; the `fs` secondary tier (`tiering/fs/manager.py`, `io.py`, `thread_pool.py`,
`async_lookup.py`) writes one file per KV chunk under `root_dir/_model_<hash>_r<rank>/...` and has NO capacity limit,
eviction or TTL (verified in v0.28.0 and v0.29.0rc1; RFC #38260 delegates eviction to each tier). On our box the tier
is a 442 GB loop-mounted filesystem; today (2026-09-03 20:38 UTC) it filled to 100% under an agentic benchmark
(354 GB in the live namespace + 89 GB in a stale namespace from another model config) and the engine started logging
`[thread_pool.py:184] Job N block I/O failed: [Errno 28] No space left on device` on every store. An external sweeper
(flan/tier-evict.sh) exists but can only delete when the engine is idle, because a file deleted between the
scheduler's lookup (HIT = `os.path.exists`) and the worker's load makes the load fail and
`vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py` asserts on job success ("we currently do not
support job failures") → engine crash. Eviction therefore belongs INSIDE the fs tier manager, which knows which keys
are in flight.

Deliver a vLLM patch that makes the fs tier self-bounding:
1. **Config** (per secondary tier entry in `kv_connector_extra_config.secondary_tiers[]`, plumbed through
   `tiering/factory.py` / `spec.py` to `FileSystemTierManager.__init__` like `n_read_threads`):
   `max_capacity_gb` (float, default 0 = unlimited = today's behaviour), `evict_high_ratio` (default 0.90),
   `evict_low_ratio` (default 0.80), `min_free_gb` (default 0; if > 0, also evict whenever `statvfs(root_dir)` free
   space drops below it — protects a shared filesystem), `evict_scope` = `namespace` (default: only this manager's
   own `_model_*_r*` directory) | `root` (all namespaces under root_dir — for a single-engine box where stale
   namespaces from previous model configs are fair game; document the multi-engine caveat).
2. **Accounting**: at startup walk the scope directory once (thousands to ~100K files: keep it O(files), no per-file
   stat storms beyond one `os.scandir` pass with `entry.stat()`), build an LRU keyed by file path with (size,
   last_access). Order the initial LRU by mtime (or atime when available). Keep `bytes_used` exact on store completion
   (+size) and on eviction/unlink (−size). Tolerate files that vanish underneath (ENOENT is not an error).
3. **LRU touch**: every lookup HIT (both the sync path and `FsAsyncLookupManager`) and every completed load moves the
   key to the MRU end. Stores insert at MRU.
4. **Safety (the crash above)**: a key that returned HIT and has not finished loading, or has a pending/in-progress
   load or store job, must NOT be evicted. Find the right place to pin/unpin: the ReqContext /
   RequestOffloadingContext hooks (`on_new_request`, `on_request_finished`, `on_schedule_end`), the lookup manager,
   `submit_load`/`get_finished_jobs`. Pins must be released on request finish/abort too (no leaks). Document the
   thread model (which thread calls lookup, which runs store/load tasks) and lock accordingly.
5. **Eviction**: triggered when `bytes_used > high` (checked after each store completion and in `on_schedule_end`),
   runs in the write-priority thread pool (never on the scheduler thread), deletes coldest unpinned files first until
   `bytes_used <= low`, in batches, unlink only (never rmdir), atomic with respect to concurrent stores of the same key
   (io.py writes `.tmp` then `os.replace`). When a store arrives and even after eviction there is no room, skip the
   store (count it) rather than hitting ENOSPC.
6. **Metrics**: expose bytes_used, files, evicted_files/bytes, skipped_stores through the existing tiering metrics
   mechanism (`tiering/metrics.py` — look at how the fs/CPU tiers register counters) so they show up on /metrics.
7. **Tests**: pytest, no GPU, under `tests/v1/kv_offload/` (find the existing fs-tier tests and extend them):
   capacity respected under a stream of stores; LRU order; pinned keys survive eviction; startup accounting from a
   pre-populated directory; ENOENT tolerance; `evict_scope=root` deletes a foreign namespace's coldest files.

Inputs (local, read-only): `TREES/v029src/` (pristine v0.29.0rc1 — the target; rc2 is identical in this area),
`TREES/v028src/` (pristine v0.28.0 — the daily's version; `tiering/fs/manager.py` and `tiering/manager.py` differ from rc1),
`TREES/v029real/` (rc1 with our unrelated attention patches applied — the diff must also apply there with `--fuzz=0`),
`REF/tier-evict.sh` (the external sweeper: its FACTS block lists what was verified in the source, and its policy).
Outputs (OUT/): `0137-fs-tier-eviction-v0290.diff` (tree-relative `a/vllm/...` paths, `patch -p1 --fuzz=0` against
v029src AND v029real), `0137-fs-tier-eviction-v0280.diff` (against v028src — only if the v0290 diff does not apply
there), `verify-0137.sh` (dry-run apply on copies of both trees + py_compile + `python -m pytest` of the new tests
against the patched tree, no GPU), `NOTES20.md` (design, thread model, exactly where pins are taken/released, config
example for our launcher: `{"type":"fs","root_dir":"/l2","n_read_threads":16,"n_write_threads":16,
"max_capacity_gb":400,"evict_scope":"root","min_free_gb":10}`, and what is VERIFIED vs HYPOTHESIS).
Rules: local files only; do not edit the input trees in place (copy to OUT/work); no behaviour change when
`max_capacity_gb` is 0; keep the patch minimal and upstreamable (this is a candidate PR — keep the style of the
surrounding code, type hints, docstrings, no flan-specific paths).
