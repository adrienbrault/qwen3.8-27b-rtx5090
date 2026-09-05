# 0142c: async-safe sync census

0142c replaces 0142, standalone against the supplied src/ tree. Do not apply it
on top of 0142; rebuild from the pre-0142 base. 0142b is unchanged. The Dockerfile
now copies 0142c, retaining the 0142 marker, offline build commands, version
assertion, compilation of all three touched files, and no enabling ENV.

## Implementation

The env parser/registry additions in `vllm/envs.py` and the installation in
`GPUModelRunner.__init__` are identical to 0142. Disabled behavior is unchanged.
`gpu/sync_census.py` adds `_Hooks.__init__` / `installed` and changes
`Census.__init__`, `scope`, `execute_step`, `sample_step`, and `report`.

`scope` registers [row, phase, explicit-call depth] by threading.get_ident().
The number of active owners is the reference count for the process-global API
patches, warning hook/filter context, and sync-debug setting. The first owner
installs them; the last owner restores them, even if that is a different thread.
A short global lock protects these transitions, never the model or output call.
A joining scope does not re-patch; a leaving scope cannot unpatch another owner.
Nested scopes on the same thread remain unsupported and raise.

API wrappers and the warnings.showwarning hook look up the current thread's
owner. Nested explicit calls and their warnings are deduplicated using that
owner's depth. Unowned threads fall through. Counts go only to the owning row;
each row is handed sequentially from execute to sample to its deferred output.
A separate Census bookkeeping lock protects pending, seen/started/finished,
the rows dictionary, runner restoration, and the report-once flag. Model calls
and logging run outside it. Each output completion is counted once, including
calls through a retained wrapper reference. Reports wait for all selected
outputs to complete successfully; failures do not produce a success table.

## Attribution and limits

The supplied traceback uses CPython 3.12's process-global warning machinery.
The shared warning context stays installed for the union of active scopes;
synchronous Python warning delivery is attributed by the emitting thread ID,
including during overlap. No overlap-specific explicit-only fallback is needed
for that machinery, so the existing report format and operation names remain
unchanged. A warning delivered on an unowned native/helper thread cannot be
attributed to a model step and is passed through, not guessed. No causal
attribution across threads is claimed. Context-aware warnings in other Python
runtimes and third-party warning-hook replacement were not validated.

If sync-debug APIs are unsupported, the existing `backend=API-fallback` proof
marks the entire census as explicit API coverage; its rows contain api:* only.
Warnings from implicit/native operations can be missed even with debug enabled.
The census counts Python API boundaries and delivered warnings, not duration,
CUDA driver calls, or all native synchronization. `api:replay` is a graph launch,
not a host sync. A completed event wait may return immediately. Overlapping
phase counts do not imply additive wall-clock stalls or removable syncs.
Unrelated threads can observe debug mode. This remains temporary diagnostic
instrumentation; monkey-patching can affect compilation guards and timing.

## Knobs, proof, and exact box gate

Use the same daily launch flags and environment as the failed run: async
scheduling ON, DFlash ns9, draft TP2, target TP2, PP=1, existing 0143 settings
unchanged. Add/pass these environment values into the served container:

```
VLLM_SM12X_SYNC_CENSUS=1
VLLM_SM12X_SYNC_CENSUS_WARMUP=20
VLLM_SM12X_SYNC_CENSUS_STEPS=50
```

The main knob accepts only absent/0 or exactly 1. Window knobs accept canonical
ASCII integers, warmup >= 0 and steps >= 1, defaulting to 20/50. Whitespace,
signs, leading zeros, non-ASCII digits and invalid values raise. PP>1,
non-DFlash (including absent speculative config), and missing AsyncOutput still
raise. Do not change knobs live.

Restart the census image and drive a c1 decode using the existing probe inside
the container (copy evidence/decode_ss.py to /tmp first):

```
python /tmp/decode_ss.py --url http://localhost:8020 --model qwen3.8-27b --conc 1 --tokens 1024 --runs 3
```

Expected per rank: one `SM12X sync census active: backend=debug+explicit` proof,
then one multiline `SM12X sync census complete: step phase count op file:line`
table after 20 decode-only warmup executions and 50 measured steps have drained.
Prefill/mixed/dummy/profile work does not advance that window. TP2 yields two
proofs and two tables across the container logs. Both ranks must stay alive;
the original concurrent runner/output exception must be absent. A fallback
backend requires treating the table as explicit-only coverage. Fewer than 70
decode steps or unconsumed outputs cannot satisfy this gate.

First inspect the count of pre-draft acceptance-read syncs (expected zero from
the source audit). The expected output event wait is a separate category.
Nonzero acceptance reads challenge the earlier source conclusion. A crash,
wrong-row attribution, missing final table after sufficient completed steps, or
duplicate table rejects this fix. For later performance work use the existing
prof_summary.py / prof_decode_split.py probes with census unset: the first
performance number remains c1 decode-only no-kernel gap, baseline 4.09 ms/step.
No speedup is predicted from this diagnostic patch.

## Verification and 0143 composition

Passed only the requested checks: `patch --dry-run -p1 --fuzz=0 --batch
--forward` on a fresh scratch copy of src/, py_compile on all three edited
Python files and test-0142.py, and nine dependency-free AST-extracted tests.
The edited scratch tree was materialized from the earlier diff for editing;
no real patch apply or old verify-0142.sh was run for this follow-up.

The barrier test drives execute/sample on the main thread and get_output on a
second thread, holds output 1 across target/sample 2, checks exact phase/op
counts (including distinct debug warnings), keeps hooks live until the last
owner leaves, and checks one report after the final output on that thread.
Other tests cover grammar, overload classification, restoration on exceptions,
debug fallback, warmup/prefill filtering, out-of-order/repeated consumption,
nested-scope rejection and unsupported configurations/missing output.

0143 is not supplied anywhere in this workdir. Its envs.py hunks therefore
cannot be inspected or stacked at fuzz 0. 0142c's envs.py hunk is identical to
0142's and touches the parser/registry boundary near base line 599; different
file claims alone cannot establish compatibility when both patches edit envs.py.
**Application on top of 0143 is unverified**, not a claimed pass. The operator
must dry-run 0142c against the actual pre-0142, post-0143 image tree at fuzz 0.
No fabricated 0143 fixture was used.

No GPU/model execution, torch/vLLM import, network, Docker build, real serving
scheduler, runtime hook writability or CUDA warning delivery was verified.
The dependency-free tests validate Python instrumentation logic only.

## Provenance

Original Apache-2.0 diagnostic code extending the supplied 0142 deliverable;
no third-party implementation or kernel copied. Integration uses the supplied
vLLM 0.29.0rc2 patched tree (Apache-2.0), Python standard-library threading /
warnings / contextlib, and existing PyTorch API interfaces. No additional
upstream commit/PR is asserted. Evidence: sync-census-async-traceback.txt,
COMMON.md, BRIEF25b.md, R186 host-sync hypothesis and the earlier NOTES25.md.
This supersedes NOTES25.md's serialized-worker/concurrency-rejection statements
for the census only; its device-acceptance audit remains unchanged.
