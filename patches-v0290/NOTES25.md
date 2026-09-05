# R190 / brief 25: census and device acceptance audit

0142 is diagnostic instrumentation. **0142b is an audit gate, not a performance
optimization.** The supplied patched V2 runner already implements the proposed
acceptance path. No removable acceptance D2H→Python→H2D round trip was found.
The COMMON.md “already done” exception applies; claiming a positive sync saving
or relocating these tensors would be misleading. Neither patch changes kernels,
tensor allocation, sampling decisions, scheduler semantics, or numerical inputs.

## Source findings (line references are to untouched src/)

- `gpu/model_runner.py:1403` (`sample`) returns sampler device counts;
  `input_batch.py:523` computes rejection counts with Triton when needed.
- `model_runner.py:1899` constructs `AsyncOutput` **before** postprocessing and
  DFlash proposal, allowing output copies to overlap the draft. It retains GPU
  source references. `async_utils.py:253` uses `.to("cpu", non_blocking=True)`.
  IDs and counts are separate asynchronous transfers, plus optional logprobs,
  masks, NaN and routed-expert outputs. They share one completion event, not one
  physical memcpy. Packing them solely to claim “one copy” adds device work and
  changes ownership without evidence of benefit; it is not done here.
- `model_runner.py:1481` / `input_batch.py:544` (`postprocess_sampled` / Triton
  `_post_update_kernel`) update last tokens, computed positions and request state
  directly on GPU. `input_batch.py:331` (`_prepare_pos_seq_lens_kernel`) reads
  the GPU computed-token state when preparing the next target batch.
- `dflash/speculator.py:331` (`propose`) receives `num_sampled`, `num_rejected`
  and `last_sampled` as device tensors. `_prepare_dflash_inputs_kernel` reads
  those counts and target positions, producing query positions, slot maps and
  selector buffers on device. Its `max().item()` at :361 is on the **CPU upper
  bound**, not an accepted-count device synchronization. Host bounds permit
  fixed padded shapes without learning acceptance on the CPU.
- `AsyncOutput.get_output` at `async_utils.py:178` performs the scheduler's event
  wait, then converts **CPU NumPy arrays** to lists. This is consumed after
  `sample_tokens` has launched the drafter. The scheduler requires token IDs and
  lengths to implement stop/finish decisions; eliminating that wait changes its
  contract. `draft_tokens_handler` also supports scheduler draft delivery; it is
  not an acceptance round trip back into DFlash.
- CPU metadata describing admissions, request order, blocks and upper bounds is
  not device acceptance state. Removing it requires scheduling/metadata-builder
  changes outside this brief. Vocab collectives and PCIe AR remain untouched.

The c1 CPU table reports 5.6 `cudaStreamSynchronize` and 2.9 `cudaGraphLaunch`
per **whole-trace** step, including prefill. Its >150 µs idle threshold gives
0.55 ms/step; this is not the decode-only 4.09 ms gap budget. No measured saving
is asserted. R186 §3.5 is a hypothesis, not proof of the suspected round trip.

## Files/functions changed

0142: `envs.py` adds `_sm12x_sync_census` and three registry entries.
`GPUModelRunner.__init__` installs instance wrappers only when enabled.
New `gpu/sync_census.py`: `_site`, `_transfer`, `install`, and `Census`
(`__init__`, `scope`, `execute_step`, `sample_step`, `report`).

0142b: `envs.py` adds `_sm12x_dflash_device_accept` and its registry entry;
`GPUModelRunner.__init__` validates DFlash/PP=1; `sample_tokens` checks that
sampled IDs/counts and persistent acceptance state share a CUDA device, and
logs the truthful zero-saving proof. No scheduler or DFlash implementation is
rewritten. Both patches apply independently or in order 0142 then 0142b.

## Knobs and logs

- `VLLM_SM12X_SYNC_CENSUS`: absent or `0` disables; exactly `1` enables.
- `VLLM_SM12X_SYNC_CENSUS_STEPS`: canonical positive ASCII decimal; default `50`.
- `VLLM_SM12X_SYNC_CENSUS_WARMUP`: canonical nonnegative ASCII decimal; default `20`.
- `VLLM_SM12X_DFLASH_DEVICE_ACCEPT`: absent or `0` disables; exactly `1` enables.

Whitespace, signs, leading zeroes, booleans, non-ASCII digits and other values
raise when evaluated. Window settings are evaluated when the census is enabled.
DFlash and PP=1 are required; concurrent instrumented runner/output invocations
raise rather than produce a falsely attributed table. Do not change knobs live.

One activation proof line per process (local `info_once` scope):
`SM12X sync census active: backend=debug+explicit`
(or `backend=API-fallback` if the sync-debug API cannot be enabled).
0142b proof, on first sampling use:
`DFlash device-side accept: 0 host syncs removed per step`

The census emits **one multiline INFO table** beginning
`SM12X sync census complete: step phase count op file:line` after all N output
objects have been consumed. Each line identifies step 1..N, phase (`target`,
`sample+draft`, `output`), count, operation, and closest vLLM source frame.
`api:replay` is a graph-launch count, **not a host sync**; totals explicitly
include replay. Other `api:*` rows count logical potentially synchronizing API
calls. `debug:*` rows count sync warnings outside those explicit calls. Nested
API/warning reports are suppressed to avoid double-counting. Zero-event steps
still receive a total row. Use phase/site rows, not the mixed total, for syncs.

Actual `InputBatch.has_prefill` excludes prefill/mixed batches. Dummy/profile
execution does not advance the window. Warmup counts decode-only executions;
only the subsequent N paired execute/sample/output steps enter the table.
A prefill execution after warmup can temporarily incur hooks but is discarded
from the census. Deferred outputs keep their own step ID even when consumed
out of order. Runner wrappers restore after the Nth proposal; pending output
wrappers restore on first consumption. Hooks, warning filters and sync-debug
mode restore in `finally` on each instrumented scope, including exceptions.
Unset census has no per-step branch, wrapper, import, tensor or kernel overhead;
only its initialization-time env lookup is added.

## Coverage and limitations

Evidence R186 identifies PyTorch 2.13.0+cu130. No torch source/runtime or usable
version pin is supplied in the launcher/NOTES21, so runtime capability detection
is used. Prefer sync-debug warnings and supplement them with explicit Tensor
item/tolist/cpu/cuda/to/copy_/nonzero, torch.nonzero/one-argument where, CUDA
Device/Stream/Event synchronize, accelerator synchronize and CUDAGraph replay
hooks. Explicit GPU-only checks exclude CPU `.item()` and CPU list conversions;
copy classification excludes D2D and nonblocking CPU↔CUDA copies.

This is a **Python boundary census, not a complete CUDA driver counter**.
Sync-debug has incomplete native/distributed coverage, and direct C++ extension
CUDA calls or launches bypassing `CUDAGraph.replay` cannot be intercepted here.
API counts describe calls that can block, not measured stalls or driver calls;
a completed event can return immediately. The fallback also misses boolean
indexing or other implicit syncs not routed through its hooked Python APIs.
Inspect the backend proof and compare with the CUDA runtime trace. Missing
trace syncs must not be interpreted as zero syncs. A frame outside vLLM is
reported explicitly. No stack is invented for native graph nodes.

Monkey-patching and warnings are process-global; unrelated threads bypass API
counting, but could observe debug mode. This is a short diagnostic window for
a serialized worker, not a production profiling mode. Wrapper installation
might interact with Dynamo guard/recompilation behavior; warm up first and
inspect graph-launch counts. An exception aborts the diagnostic rather than
emitting an incomplete success table. Fewer than N consumed decode outputs
produce no final table. Neither runtime hook writability nor graph capture
compatibility could be verified offline.

## Tensor lifetime / replay / preemption audit

**No tensor lifetime is changed by either patch.** Existing ownership is:

| Tensor/state | Existing owner and replay safety |
| --- | --- |
| `num_sampled`, `num_rejected`, sampled IDs | Per-step sampler tensors; eager postprocess and preparation consume them on the main stream. They are not bound as changing graph-input addresses. AsyncOutput retains copies' device sources until output consumption. |
| `req_states.num_computed_tokens.gpu`, last sampled tokens, draft tokens | Persistent request-slot storage; Triton updates in place. Next-target preparation reads this storage on the same stream. Admission stages initialization and finish/removal handles slot reuse. |
| Target `InputBuffers.positions`, query starts, slot maps | Persistent buffers, filled for the current mapping before forward; no replacement addresses introduced. |
| DFlash query positions/input IDs/seq lengths/query starts and shared slot maps | Persistent InputBuffers/BlockTables allocations, populated before graph replay on the main stream. |
| DFlash `sample_indices`, `sample_pos`, `sample_idx_mapping` | Persistent allocations in DFlashSpeculator.__init__; preparation rewrites active rows and resets padding (mapping -1). Full graph reads stable addresses. |
| DFlash context positions, context slot mappings, hidden state staging | Persistent allocations; eager context-KV preprocessing precedes draft replay. Rejected/unresident slots use PAD_SLOT_ID. |
| CPU output IDs/counts and completion event | Per-output ownership; one completion event gates CPU access. No newly shared pinned storage or earlier source release. |

Preemption, finish mid-block and admission continue to use the original
request→slot mapping, staged writes, masks and scheduler logic. No accepted
suffix is cached across steps by these patches. The census retains only Python
Counters and output wrappers; it neither retains extra GPU tensor references
nor changes reuse ordering. These are source-based safety observations, not a
GPU fidelity result.

## Build, verify, measure

Build with `deliver/` as the Docker context; e.g.
`docker build -f deliver/Dockerfile.dflash-sync-census -t vllm-sync-census deliver`.
For the audit layer use its Dockerfile, optionally `--build-arg BASE=vllm-sync-census`.
Build RUN steps have `--network=none`, assert vLLM 0.29, apply at fuzz zero,
compile every touched Python file and write the required marker. They set no
enabling ENV. The base image must already be local; Docker builds were not run.

`bash deliver/verify-0142.sh` performs standalone dry-run AND real applies with
exit-code checks, py_compile, a stacked apply, and stdlib-only AST-extracted
helper tests with fake CUDA objects. It imports neither torch nor vLLM. Seven
tests cover invalid grammar, copy overloads, nested-call deduplication, restore
on exceptions, fallback, concurrency rejection, warmup/prefill exclusion and
out-of-order output consumption. These are instrumentation logic tests, not
CUDA correctness tests.

Operator sequence (all Python probes inside the served container; copy the
existing evidence probes into `/tmp` if necessary):

1. Baseline unchanged launcher, c1, then c8/c16. Restart separately with census
   enabled and default 20/50 window for each concurrency. Use
   `python /tmp/decode_ss.py --url http://localhost:8020 --model qwen3.8-27b --conc 1 --tokens 1024 --runs 3`
   to drive sufficient decode; use separate c8 and c16 runs. Save both ranks'
   activation lines and tables from docker logs. This run's throughput is invalid
   for comparing speed. Inspect the earliest pre-draft sync site, and distinguish
   the expected output event wait from acceptance reads (expected zero).
2. With census **unset**, reproduce R183 torch tracing using the operator's
   existing capture recipe; run `prof_summary.py` and
   `prof_decode_split.py /tmp/rank0.<timestamp>.pt.trace.json.gz`, then rank1.
   Do not pass benchy.json. The split's last-NCCL heuristic is suitable only for
   the fixed prefill-then-decode recipe; compare with census phase classification.
3. Run the existing **bf16 decode-fidelity ruler and greedy-equivalence gate**
   before judging any throughput change: identical prompts/seeds/token IDs,
   compare accepted lengths and per-position results, long (~30K) contexts,
   c1/c8/c16, variable admissions, cancellations, preemption, finish mid-block,
   and slot reuse. The ruler executable is not included in this source dump;
   use the operator's existing gate, not the stochastic decode_ss workload as
   a substitute. Compare 0142b on/off; any token or acceptance mismatch rejects
   this audit layer immediately. Repeat for census after it disables itself.
4. First performance number: **c1 decode-only no-kernel gap ms/step**, baseline
   **4.09 ms**. Expect no reduction from 0142b, which removes zero syncs; any
   statistically significant speedup needs attribution, not promotion as an
   acceptance optimization. The acceptance-round-trip hypothesis is falsified
   if the census/trace shows no acceptance read before the draft (as source
   predicts). Native unobserved syncs require deeper CUDA tracing, not inferred
   savings. Keep AR/vocab categories out of this brief's claimed budget.

Offline result: standalone and stacked zero-fuzz applies passed; all touched
Python files compiled; seven dependency-free tests passed. No GPU, network,
serving-box access, torch/vLLM import, Docker build, runtime census, memory-race
validation or fidelity ruler was available/performed.

## Provenance for THIRD_PARTY.md

Original diagnostic/audit code written for this brief; SPDX Apache-2.0.
Source integration derives from the supplied vLLM 0.29.0rc2 patched tree
(Apache-2.0, existing headers retained), including local 0129 DFlash graph work.
No external implementation, PyTorch source, new PR, or third-party kernel was
copied. Runtime APIs are PyTorch sync-debug/warnings/Tensor/CUDAGraph interfaces;
Python warnings/contextlib are standard-library APIs. Evidence references:
R186 §3.5, NOTES21, launch-daily.sh, r183 c1 CPU table and prof_decode_split.py.
The exact upstream commit of the dumped vLLM tree is not supplied; no commit
identity beyond the supplied image/version is asserted.
