# BRIEF20 design and verification notes

## Design

FileSystemTierManager scans either its model/rank namespace or root_dir once
at startup. It visits directories with os.scandir, calls DirEntry.stat once
per regular .bin file, and initializes an OrderedDict LRU from the greater of
mtime and atime. The LRU value is (size, last_access), and bytes_used is
updated on completed stores, observed disappearance, and eviction.

The configuration is passed by SecondaryTierFactory's existing generic
keyword forwarding. TieringOffloadingSpec documents the fs-specific options,
and FileSystemTierManager validates and consumes max_capacity_gb,
evict_high_ratio, evict_low_ratio, min_free_gb, and evict_scope. A zero
max_capacity_gb with the default zero min_free_gb never evicts or skips a
store.

The scheduler thread calls the tier lifecycle, lookup submission/result
handling, and job submission/result handling. FsAsyncLookupManager has one
lookup worker. DualQueueThreadPool workers perform loads, stores, and eviction;
periodic eviction is placed on its write-priority queue and never unlinks on
the scheduler thread. A cache lock protects the LRU, byte accounting,
reservations, pins, and metric counters. Slow lookup probing happens outside
the lock under temporary pins; eviction holds the lock for at most one
128-candidate unlink batch before releasing it.

Pins are taken and released at these exact points:

- Before an async existence probe, every candidate path gets a temporary pin.
  Hits atomically touch the LRU and transfer to an idempotent per-request pin;
  misses release the temporary pin. The scheduler-side cached HIT path also
  touches and pins, covering both initial and repeated HITs.
- submit_load and submit_store take independent job pins before enqueueing.
  get_finished_jobs releases them only after the pool publishes completion.
- A completed load touches each successfully loaded path. Collection of that
  completion releases both its load-job pin and that request's pins for the
  loaded paths.
- on_request_finished removes the request from the active set and releases all
  remaining request pins. A lookup worker finishing after an abort sees the
  request inactive and cannot recreate a request pin.

Stores reserve their prospective bytes before I/O, preventing parallel stores
from overcommitting or double-accounting the same path. Preflight eviction
runs in the worker when the hard maximum or minimum-free reserve would be
violated. If unpinned files cannot make enough room, the store becomes a
successful no-op and increments skipped_stores, avoiding an ENOSPC job
failure. Atomic file publication remains the existing temporary-file plus
os.replace implementation in fs/io.py.

After each store completion, the worker evicts cold unpinned files when
accounted plus reserved bytes exceed the high watermark, stopping at the low
watermark. on_schedule_end enqueues a deduplicated write-queue eviction check;
with min_free_gb configured, it checks statvfs even below the capacity
watermark. Eviction only unlinks regular files already recorded by the startup
scan or later accounting; it never removes directories. ENOENT forgets the
record and subtracts its accounted size without treating it as an error.

evict_scope=namespace is the safe default. evict_scope=root includes .bin files
from foreign model/rank namespaces and is intended only when one engine owns
the root. Multiple engines cannot share in-memory pins, so root scope can
delete another engine's in-flight files.

The tier registers and emits gauges for bytes_used and files, plus counters for
evicted_files, evicted_bytes, and skipped_stores through the existing tier
metric-definition and get_stats aggregation hooks.

Launcher example requested by the brief:

    {"type":"fs","root_dir":"/l2","n_read_threads":16,"n_write_threads":16,
    "max_capacity_gb":400,"evict_scope":"root","min_free_gb":10}

## Variant decision

Only OUT/0137-fs-tier-eviction-v0290.diff is needed. The v0.28 difference in
fs/manager.py is confined to the pre-existing cross-process-sharing docstring,
outside every changed hunk. The same patch applies with zero fuzz to v0.29
src, v0.29 real, and v0.28 src. Therefore
OUT/0137-fs-tier-eviction-v0280.diff is intentionally not created.

## Verification performed

Command executed:

    OUT/verify-0137.sh

Fresh-copy workspace from the successful run:

    OUT/work/verify-0137.krAhbA

Literal outcomes:

- v0.29 src patch dry-run with --fuzz=0: PASS (5 files patched).
- v0.29 src actual apply with --fuzz=0: PASS.
- v0.29 src py_compile of all 5 touched files: PASS.
- v0.29 src pytest: PASS, 9 passed, 0 failed, 1 warning.
- v0.29 real patch dry-run with --fuzz=0: PASS (5 files patched).
- v0.29 real actual apply with --fuzz=0: PASS.
- v0.29 real py_compile of all 5 touched files: PASS.
- v0.29 real pytest: PASS, 9 passed, 0 failed, 1 warning.
- v0.28 src patch dry-run with --fuzz=0: PASS (5 files patched).
- v0.28 src actual apply with --fuzz=0: PASS.
- v0.28 src py_compile of all 5 touched files: PASS.
- v0.28 src pytest: PASS, 9 passed, 0 failed, 1 warning.
- Total focused pytest result: PASS, 27 passed, 0 failed.
- Ruff formatting check and Ruff lint for all touched files: PASS.

The warning on each pytest run is the source archive's expected missing
generated vllm._version module. This host has no complete vLLM development
environment, so verify-0137.sh used an existing local pytest/numpy Python,
a cached local CPU Torch build, and the OUT/work/test-bootstrap import shim
for unrelated distributed import-time dependencies. No network, GPU, remote
host, or input-tree mutation was used.

## VERIFIED versus HYPOTHESIS

VERIFIED:

- Source inspection confirms the async lookup worker, scheduler-owned lookup
  state, generic factory keyword forwarding, dual read/write queues, atomic
  temporary-file replacement, and tier get_stats aggregation hooks described
  above.
- Tests exercise streaming capacity, LRU touch ordering, request and job pin
  release, pinned-store skipping, startup accounting, ENOENT tolerance,
  min-free eviction, root-scope foreign eviction, metrics, and factory config
  forwarding.
- Zero-fuzz application, compilation, and all focused tests passed on all
  three requested source variants.

HYPOTHESIS / not integration-tested here:

- The no-GPU unit tests do not run a full vLLM engine or reproduce production
  scheduler timing under agentic load.
- statvfs behavior was tested deterministically with a fake low-free result;
  live filesystem accounting under unrelated concurrent writers was not
  stress-tested.
- Root-scope safety assumes one engine owns root_dir, as documented; there is
  deliberately no cross-process pin coordination.
