# BRIEF25b — 0142 sync census must survive async scheduling (follow-up to BRIEF25)

Read COMMON.md first (same rules: verify only with patch --dry-run + py_compile + your dependency-free tests, no GPU/model execution, one pass).

## What happened on the box (evidence/sync-census-async-traceback.txt, evidence/engine-diag-full.log)
Image = daily + 0142 + 0143, booted with the daily flags (async scheduling ON, DFlash ns9, TP2, PP=1) and
VLLM_SM12X_SYNC_CENSUS=1. Proof line appeared (`SM12X sync census active: backend=debug+explicit`). The first decode request killed
the engine on both ranks:

    File ".../vllm/v1/worker/gpu/sync_census.py", line 167, in get_output
    File ".../vllm/v1/worker/gpu/sync_census.py", line 67, in scope
    RuntimeError: SYNC_CENSUS does not support concurrent runner/output calls

Under async scheduling the runner's execute_model for step N+1 runs on the main worker thread while AsyncOutput.get_output for step N
is drained on the executor's output thread; `Census.scope` takes a single process-wide non-blocking `_LOCK`, so the overlap that IS
the production regime raises. The census is only useful in that regime (the daily runs async ON; `--no-async-scheduling` changes the
very sync structure being measured), so the guard has to go, not the overlap.

## Deliverable
`deliver/0142c-dflash-sync-census-v0290.diff`: a replacement for 0142 (standalone against the same src/ base, same envs, same proof
line, same report format) whose `scope` is safe when the "output" phase of step N overlaps the "target"/"sample+draft" phase of
step N+1 on another thread. Requirements:
- No exclusive lock across phases. Ownership is per thread already (`owner = threading.get_ident()` and the wrapped APIs fall
  through for other threads); keep that and make the shared, process-global pieces (torch sync-debug mode, the patched API
  attributes) reference-counted so the second concurrent scope neither re-patches nor un-patches under the first. Counts must
  attribute to the row of the thread that owns each scope, never to the other thread's row.
- sync-debug-mode warnings: if they are captured through a warnings hook / logging handler, attribute them by thread ident to the
  owning scope; if that cannot be done reliably, document the fallback (count only explicit API calls in the overlapped phase) and
  mark those rows in the report.
- The rows dict, `pending`, `started/finished` counters and the un-install at `started == steps` are touched from two threads:
  make them safe (a small lock for bookkeeping only, never held across the model call).
- `report()` must still fire once, after the last output phase completes, even if the last get_output runs on the output thread.
- Keep the fail-loud behaviour for the truly unsupported cases (PP>1, non-DFlash, missing AsyncOutput).
- Update deliver/test-0142.py: add a test that drives execute/sample on one thread and get_output on a second thread with an
  overlap (barrier), asserting no exception, correct per-row attribution, and that the report fires once.
- NOTES25b.md: what changed in scope/bookkeeping, what the overlapped attribution can and cannot say, and the exact reproduction
  gate for the box (same env as before; expected: proof line, then the `SM12X sync census complete` table after 20 warm-up + 50
  steps of a c1 decode). Also state whether 0142c still applies on top of 0143 at fuzz 0 (they touched different files last time;
  check envs.py).
Dockerfile.dflash-sync-census updated to apply 0142c. Keep 0142b as is.
