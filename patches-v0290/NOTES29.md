# 0146 — exact target sequence-length snapshot (partial tier 1)

**The requested one-copy target-plus-draft tier 1 is impossible on this tree.**
This deliverable uses COMMON.md's source-impossibility exception: consolidate
five target-phase reads into one early asynchronous copy and one event wait;
keep the draft's separate exact copy. Expected census total **9 → 5**, not 4.
No 0146b is delivered. Neither tier's requested all-six sharing/zero-copy claim
is made. The historical filename and knob are retained, with their narrowed
meaning stated here and in the activation line.

## Why target and draft cannot share a snapshot

`gpu/input_batch.py:_prepare_pos_seq_lens_kernel` writes target sequence length
`T = num_computed_tokens + query_len`, including all verification tokens.
`gpu/model_runner.py:prepare_inputs` launches that kernel before target metadata
and forward. Rejection counts for this verification do not exist yet.

After target forward and sampling, `dflash/speculator.py:propose` calls
`prepare_dflash_inputs`. `_prepare_dflash_inputs_kernel` computes
`valid_ctx_end = ctx_end - num_rejected`, loads `last_valid_pos` from target
positions, and writes its **own** `input_buffers.seq_lens` as
`min(last_valid_pos + 1 + num_query_per_req, max_model_len)`.
For contiguous positions this is `min(T - rejected + Q, limit)`. Example:
T=128, rejected=1, Q=10 gives draft length 137, not 128. At page size 64 this
also changes the number of pages from 2 to 3. Even zero rejection generally
adds Q. Padding is independently reset to zero by the draft preparation kernel.
The tensors differ in address, generation, values, and potentially padded size.

DFlash `_build_draft_attn_metadata` delegates to `DraftModelSpeculator` in
`gpu/spec_decode/speculator.py`. That method passes the draft InputBuffers
sequence lengths to `build_attn_metadata`. The draft FlashInfer builder reads
the deprecated property for native planning. The target snapshot cannot supply
that read. Copying after rejection would be too late for target planning;
computing draft lengths from a CPU upper bound loses rejection information.
This patch leaves the entire sample/draft phase unchanged.

## Callers and multiplicity

All accesses to this CommonAttentionMetadata property in executable source:

- `attention/backends/flashinfer.py:FlashInferMetadataBuilder.build`: guarded
  read around :1746, followed by NumPy page counts, `_compute_flashinfer_kv_metadata`
  (paged indptr, last-page lengths and GPU indices), native wrapper `plan()`.
- `attention/backends/utils.py:make_local_attention_virtual_batches` around :481:
  exact lengths determine virtual chunk boundaries. Its caller is
  `model_executor/layers/attention/chunked_local_attention.py`. This is not the
  served Qwen GDN metadata/split path.
- `attention/backend.py:CommonAttentionMetadata.num_computed_tokens_cpu`:
  lazy host subtraction of query lengths; its cache now receives the same
  shared exact target snapshot in the enabled target build.
- `attention/backends/mla/rocm_aiter_mla_sparse.py` around :555/:559 also reads
  the property. This ROCm MLA path is outside this SM120/Qwen configuration.
  `.orig` files are backups. Pooling, TurboQuant and legacy input-batch fields
  with the same name are different data structures, not this property.

`split_decodes_and_prefills` uses CPU query starts/max query length and,
optionally, CPU is_prefilling; it does not read exact CPU sequence lengths.
`gdn_attn.py:GDNAttentionMetadataBuilder.build` already uses device sequence/state
information and CPU scheduling fields. Mamba's host prior-state/prefill logic
uses `seq_lens_cpu_upper_bound`; it does not cause these deprecated-property
copies. There is no safe new host read to eliminate in those helpers here.

`mamba_hybrid.py:prepare_attn` calls `attn_utils.build_attn_metadata` once. The
latter constructs a NEW CommonAttentionMetadata per KV-cache group; the lazy
CPU cache was therefore per group, not per batch. Its inner attention-group
loop shares only within that KV group. The runner's `init_attn_backend` call
has no active-layer filter, so it includes draft attention groups; the draft's
separate initialization DOES filter to draft layer names. Target-layer hidden
state taps are outputs, not five independent metadata builds.

`core/kv_cache_utils.py` groups compatible specs and splits buckets using the
smallest layer bucket. With 16 target full-attention, 48 GDN, and 5 distinct
draft sliding-attention layers, bucket size 5 gives 4 full-attention groups,
10 GDN groups, and 1 draft attention group. Four plus one FlashInfer groups
explains the five target-phase copies; GDN groups add none. The census records
the property site, not group IDs, so this attribution is source-derived. The
proof prints the actual number of FlashInfer-bearing KV groups receiving the
snapshot (expected 5), rather than hard-coding a layer-tap count.

## Why tier 2 is not delivered

The served launcher explicitly sets **VLLM_SM12X_NVFP4_XQA=0**. In
FlashInferMetadataBuilder.__init__, this disables the SM120 NVFP4 TRTLLM/XQA
decode route. Spec verification at query length 10 is handled by native FA2
prefill wrappers, even on a scheduler decode-only step. The draft also uses
native planning for its non-causal attention. FULL CUDA graphs do not remove
planning: metadata is rebuilt/staged before every replay.

FlashInfer's `build` already skips host lengths when `all_uses_trtllm` is true.
There is nothing additional to remove on that route under this brief. On the
served native route, replacing exact host lengths with optimistic lengths
changes page lists and last-page masks. The native wrapper's plan/forward
contract here does not supply a separate exact device length to repair those
host-derived masks. Changing to XQA, speculative-XQA, or a different planning
implementation would change kernels and require a separate fidelity study.

## Implementation and ownership

Changed functions/classes (six Python files, patch under 400 lines):

- `envs.py`: new strict `_sm12x_seqlens_one_copy` parser, typing declaration,
  and registry entry, placed to coexist with the supplied 0142c patch.
- `GPUModelRunner.__init__`: validate supported configuration and allocate
  one persistent pinned int32 staging buffer when enabled.
- `GPUModelRunner.prepare_inputs`: immediately after `prepare_pos_seq_lens`,
  enqueue nonblocking D2H on that same current CUDA stream and record an event;
  attach its snapshot to this InputBatch.
- `InputBatch`: optional snapshot field, None for existing/dummy constructors.
- `MambaHybridModelState.prepare_attn`: pass the snapshot into the shared build.
- `attn_utils.build_attn_metadata`: validate supported builders; wait once before
  the first metadata builder, slice for its padded/eager request count, compute
  host computed-token counts once, and populate both private CPU caches of
  every CommonAttentionMetadata with the same tensors.
- New `seqlens_copy.py`: `SeqLensCopy.__init__/begin` manage persistent staging
  and reject unconsumed reuse/capture/invalid tensors. `SeqLensSnapshot.__init__/get`
  own transfer references and the event, wait under gpu_sync_allowed, and clone
  completed host staging to independent CPU storage before publication.

The CPU clone is deliberate: old metadata may survive a step or feed native
planning, so its storage must not alias staging overwritten on the next step.
This adds a small CPU allocation/copy, not another D2H. All target groups share
that clone and one computed-token tensor. No device tensors or graph-bound
addresses are replaced. The snapshot retains the device source until the wait;
the runner also owns the persistent device input buffer.

The writing kernel, D2H and event are ordered on the same current stream. The
wait therefore completes the exact copy before ANY target builder can read it;
it does not wait for later-enqueued target preparation kernels. There is no
side-stream dependency to guess. Real metadata construction precedes forward
and graph replay; the copy is never captured. Dummy/profile/capture batches
have no snapshot and use the original path. Replays retain the original
persistent target/draft inputs. Draft graph buffers are independently rewritten
after sampling, using the original preparation and host-copy code.

Preemption, admission, finish mid-block and slot reuse continue through the
original idx_mapping and GPU computed-token state. The snapshot is made anew
for each real batch, after that mapping has been applied; no request IDs,
accepted suffix, or CPU upper bound are reused as exact lengths. A subsequent
step cannot overwrite staging until the previous snapshot was consumed; an
unexpected lifecycle raises instead of silently returning stale data. These
are source-ordering arguments, not runtime race/fidelity results.

## Knob, proof and build

`VLLM_SM12X_SEQLENS_ONE_COPY`: unset or exact `0` disables; exact `1` enables.
Anything else, including blanks, whitespace, signs, `01` or boolean words,
raises on runner initialization. Do not change the environment mid-process.
Supported integration: V2 runner, Qwen3.5 model_type, DFlash, fixed verification,
PP=DP=DCP=PCP=1; TP2 is supported. Only FlashInfer/GDN builders are accepted on
the enabled target build; DCP's in-place host-length adjustment is rejected.
Unsupported configurations/tensors and an attempt to copy inside CUDA capture
raise. The knob belongs to this V2 integration; do not use the legacy runner.
Unset adds Python plumbing only: same kernels, device/host tensor allocation
paths, numerical inputs and original lazy properties; no snapshot is allocated.

One info_once line per process, at first snapshot consumption (expected n=5):

    SM12X seq_lens one-copy active: 5 target seq_lens_cpu consumers share one D2H per step; draft keeps its own copy

The qualified line intentionally does not falsely promise six identical lengths.
`docker logs` should show two rank-local copies. The count is the number of
FlashInfer-bearing target-build KV groups supplied with the CPU cache; it is
not a runtime counter of calls to the property.

Build with `deliver/` as context:
`docker build --network=none -f deliver/Dockerfile.syncfree-seqlens -t vllm-seqlens0146 deliver`.
The local base image must exist. RUN has no network; the build asserts vLLM
0.29, applies at fuzz zero, compiles all touched files, writes
`/opt/prs-markers/0146`, and enables no knob. The version import is box/build-only.
Use the launcher's EXP environment passthrough to add
`-e VLLM_SM12X_SEQLENS_ONE_COPY=1`; keep all other launch settings fixed.

## Verification and measurement

`bash deliver/verify-0146.sh` copies src/vllm to .work/0146/vllm, dry-runs and
really applies at fuzz zero with exit-status checks, compiles all six files,
and runs five stdlib-only tests. Tests AST-extract changed code without importing
torch/vLLM; fake deferred DMA checks wait-before-read, one wait, shared computed
values, padding, safe old-snapshot lifetime, invalid grammar/tensors/capture,
and disabled behavior. A shape counterexample checks target/draft mismatch and
bucket math. A separate scratch copy also dry-runs/really applies 0142c on top
and compiles its touched files. All passed; see verify output/CODEX-final.md.

Operator sequence (probes run INSIDE the container, e.g. via docker exec):

1. Layer the supplied `r190d/0142c-dflash-sync-census-v0290.diff` on 0146, enable
   `VLLM_SM12X_SYNC_CENSUS=1`, and compare knob off/on at c1, then c8/c16.
   Keep its default 20 warmup/50 measured decode-only steps. Expected table:

   | Phase/site | Baseline | 0146 |
   | --- | ---: | ---: |
   | target backend.py:473 api:to | 5 | 0 |
   | target seqlens_copy.py api:synchronize | 0 | 1 |
   | sample+draft backend.py:473 api:to | 1 | 1 |
   | target + draft api:replay | 2 | 2 |
   | output event api:synchronize | 1 | 1 |
   | mixed census total | 9 | 5 |

   0142c deliberately excludes nonblocking transfers. Thus global api:to=1
   does NOT mean one physical D2H: this patch performs TWO sequence-length
   transfers per target+draft cycle, one asynchronous target and one original
   blocking draft. Event waits are counted explicitly. Replays are API calls,
   not proven host synchronization. Check CUDA trace as well; the census is
   not a CUDA-driver transfer counter. Tier 2's hypothetical zero reads is not
   an expected result for this deliverable.
2. With census unset, run the existing bf16 decode-fidelity ruler at ctx 0 and
   30K against a same-compile-artifact control, matching prompts/seeds/weights,
   fixed flags and graph captures. Required: bitwise **20/20, median 0** for
   patched-on versus patched-off outputs, with identical accepted lengths.
   Include c1/c8/c16, admissions, padding, cancellations, preemption and finish
   mid-block. Any mismatch rejects the patch. The ruler executable is absent
   here; use the operator's existing gate, not decode_ss as a substitute.
3. Run the supplied probes/decode_ss.py (copied as /tmp/decode_ss.py):
   `python /tmp/decode_ss.py --url http://localhost:8020 --model qwen3.8-27b --conc 1 --tokens 1024 --runs 3`
   Select the operator's code workload consistently; repeat with --conc 8 and
   --conc 16. Confirm steps/s using target-step counts/acceptance, since emitted
   tokens/s alone can hide acceptance drift. Profile separately with the existing
   R183 recipe and prof_summary.py / prof_decode_split.py, both ranks.

**First number: c1 decode-only no-kernel gap, baseline 4.09 ms/step.** Expected
direction is down, with c1 steps/s up if these waits exposed GPU idle time.
Do not claim a speedup. Even removing the entire 4.09 ms of an 18.4 ms step
would only reach 14.31 ms (~28.6% more steps/s); this partial consolidation has
a lower practical ceiling and retains both a target and a draft wait. If the
four removed calls are confirmed but c1 step/gap does not move, those repeated
waits were not the critical gap; investigate launch overhead/CUDA-graph coverage.
That result cannot establish that the remaining draft wait is cost-free.

Risks: native planner may have undocumented host-buffer lifetime/mutation rules;
CPU clone and event creation may outweigh savings for short contexts; earliest
copy on the main stream can delay following kernels; residual draft and output
waits remain. More asynchronous host progress must be checked against full-graph
buffer staging under real workload. No GPU execution, torch/vLLM import, Docker
build, runtime census, native FI inspection, fidelity ruler, race test, or
throughput measurement was possible offline. No src/evidence files were edited.

## Provenance for THIRD_PARTY.md

Original patch/helper/test code written for this brief, Apache-2.0. Integration
uses the supplied Apache-2.0 vLLM 0.29.0rc2 patched source tree and its existing
0129 DFlash graph path; original headers retained. Technique: standard PyTorch
pinned-host nonblocking copy plus CUDA event ordering and CPU-owned snapshot.
No external implementation, kernel, PR or code was fetched/copied. The supplied
image/version is the provenance pin; no upstream commit hash was supplied.
Audit evidence: launch-daily.sh, target/drafter configs, r190d census + 0142c +
NOTES25, R186/R183 timing budget. No new FlashInfer/PyTorch source is vendored.

## Operator addendum (2026-09-05 07:55 UTC)

- Verified in the served image (not offline): dry-run and real apply at fuzz 0 (6 files), py_compile of all six touched files, `test-0146.py` 5/5 against the applied site-packages tree.
- One extra hunk added by the operator (envs.py, `compile_factors` ignored set): `VLLM_SM12X_SEQLENS_ONE_COPY` is excluded from the torch.compile cache key. The knob changes host-side metadata plumbing only (no traced-graph change, as this note states), and R193 showed that two fresh compile artifacts of one configuration differ at T=0 (inductor's runtime Triton autotune choices are bundled into the artifact). Without the exclusion, knob ON and OFF would be different artifacts and the bitwise ruler in step 2 could never pass or fail for the right reason; with it, ON loads the artifact OFF saved.
- Images: `vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-syncfree` (BASE pcieipc) and `...-pcieipc-r190diag-syncfree` (BASE r190diag = 0142c census + 0143, for step 1). Unit: `flan/r194-syncfree.sh`.
