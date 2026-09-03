# NOTES16 — vLLM v0.29.0rc1 NVFP4-KV / DFlash2 rebase

## Result and scope

The phase-1 chain was mechanically rebased from v0.28.0 onto the supplied v0.29.0rc1 tree. No patch was dropped. Phase 1 verified before work began on phase 2. The two phase-2 ports are separate tree-root diffs and also verify on top of the complete phase-1 chain.

Verification here is intentionally limited to exact-context patch application and Python parsing. No runtime, model, GPU, build, test, SSH, or container command was run. Every dry-run used `-p1 --fuzz=0`; a real local application immediately followed each successful dry-run only to construct the input state for the next dry-run and for `py_compile`.

## Upstream reconciliation

- CUDA-graph reserve (#53306/#53955/#53682): v0.29's V2 runner now profiles graph memory against a minimal KV cache in a throwaway graph pool before sizing the real KV cache. That removes the old ledger's general premise that the KV pool is always sized before capture with zero graph reserve. The profiling capture reaches the metadata builders and therefore should observe allocations made by the pooled FlashInfer wrappers, including their integer workspaces, subject to v0.29's FULL-graph sampling/extrapolation and the exact captured shape set. This is not a proof that every custom pool shape is estimated exactly. `0131` remains useful because it reduces real persistent memory and the corresponding profiled reserve from 8 MiB to 1 MiB per pooled wrapper.
- Uniform capture sizes (#50488/#54418): v0.29 adds request-count-derived uniform decode sizes but caps default coverage at the platform-safe capture ceiling. The local causal pool remains keyed by request count and guarded by `_prefill_cudagraph_max_bs`; `0119` makes its width agree with the final uniform decode width. The non-causal DFlash pool is also keyed by request count, and `0129` still caps admitted DFlash graph request counts. Therefore the new default list changes which graph descriptors may reach the pool, not the pool key or wrapper shape contract. Sizes beyond the cap remain eager.
- XQA head-dimension fallback (#53111): the upstream per-KV-group head-dimension fallback is retained. `0103` applies the NVFP4 env gate after eligibility is known, and only disables TRTLLM/XQA when `VLLM_SM12X_NVFP4_XQA=0` or the group is unsupported.
- DFlash RoPE layout (#54373): upstream `is_neox_style` derivation and validation remain intact. `0107` only adds ModelOpt/W4A16 dense-weight materialization and defers the fused-KV build until post-load repacking when necessary.
- DFlash2 selector/loading (#52816/#53797/#53435): upstream supplied the DFlash2 class, local convolution/candidate selector, speculators-format mapper, and corrected load flow. It did not absorb `0106`'s temperature-zero deterministic path, explicit fp32/fp64 Gumbel arithmetic, or the final index clamp. Those guards survive.
- V2 default (#53183): `0104` remains byte-for-byte applicable but modifies V1 `GPUModelRunner`/`LLMBaseProposer`. It is dead on the current V2 daily unless the runner falls back or is explicitly selected; it is retained for chain fidelity.
- Mamba/GDN work (#52789/#53877/#52539/#53077): upstream internal prefill checkpoints, packed-decode FP32 beta, Qwen head-ratio handling, and empty-draft reset are preserved. `0111` still contributes the opt-in ReplaySSM speculative state layout, cursors/rings, commit/reset metadata, and GDN kernel. Its metadata branch now classifies prefill/decode from `is_prefilling` and leaves upstream's zero-draft fallback unchanged when ReplaySSM-spec is off.
- Attention metadata/LSE (#53002/#53336/#52796): v0.29's physical `KVCacheLayout` API replaces the removed string helper, so rebased wrappers use `get_flashinfer_layout_string(self.kv_cache_layout)`. Upstream LSE normalization and merge code is untouched; the non-causal FA2 path only changes wrapper choice and dtype/cache views.
- Deprecated parameters (#53559/#52557): the removed `use_prefill_decode_attention` gate is not restored. `0116`'s tp=1 invalid-token mask is moved into the new tp=1 fast return in `VocabParallelEmbedding.forward`.

## Patch ledger and hunk rationale

### 0101-sm120-nvfp4kv-fa2-routing-v0280.diff

Verdict: **rebased; semantics-changed-needs-metal-check**.

Hunks, in order: import `cache`; admit SM120 NVFP4; constrain SM120 NVFP4 to HND layouts through the new `KVCacheLayout` API; derive `use_fa2_nvfp4_kv`; allocate the graph-bound prefill pool and uniform width; override upstream decode selection only for the FA2 fallback; allocate persistent QO indptr; select model Q dtype; extend the wrapper accessor signature; construct a graph-bound prefill wrapper; select FA2 versus trtllm-gen for ordinary prefill; select FA2 versus trtllm-gen for decode; expose uint8 NVFP4 metadata; recognize uniform multi-token decode; plan uint8 KV; choose model output dtype for FA2; pass packed NVFP4 views to prefill; choose model output dtype in the second prefill route; pass uint8 KV there; allow native decode for the FA2 route; size its output path; split packed data/scales; avoid FP8 output conversion on SM120 FA2; route stores through the cached linear-V-scale overlay; initialize and log the overlay in the worker.

Failed/moved hunks: the old string-returning cache-layout hook became `supported_kv_cache_layouts`; decode selection gained upstream's per-group XQA head-dimension fallback; page buffers became `CpuGpuBuffer`; and both wrapper constructors now use `get_flashinfer_layout_string`. All five sites were moved to those rc1 contracts. The isolated failure count is five; the earlier partial-chain log's smaller count reflects different already-applied context.

### 0103-sm120-nvfp4-xqa-decode-v0280.diff

Verdict: **rebased; semantics-changed-needs-metal-check**.

Hunks: cache/log `VLLM_SM12X_NVFP4_XQA`; derive eligible NVFP4 XQA use; retain FA2 when off; keep the generic/dedicated distinction; allocate graph-stable scale scratch metadata; compact scale pages in the global workspace; keep dedicated XQA off for this original generic route; prepare compact scales; avoid FP8 output for XQA; pass compact scales.

Failed/moved hunks: the FA2 override moved after upstream's head-dimension fallback; `use_dedicated_xqa` gained upstream routing conditions and retains the NVFP4 exclusion until 0132's masked route; the call-site scale variable is now `decode_kv_block_scales` after workspace preparation.

### 0104-mtp-drafter-full-cudagraph-v0280.diff

Verdict: **verbatim**.

Hunks: advertise FULL support for the narrowed proposer; construct padded graph descriptors; carry padded token/sequence counts; make replay dispatch shape-stable; retain descriptor buffers; capture supported sizes; select exact/padded descriptors; bind stable input/output buffers; run the proposer through FULL replay; and preserve eager fallback/hidden-state extraction. It applies unchanged but is V1-only under the V2 default.

### 0105-dflash-noncausal-nvfp4-fa2-v0280.diff

Verdict: **rebased; semantics-changed-needs-metal-check**.

Hunks: replace the blanket non-causal NVFP4 rejection with an FA2 backend selection while retaining the sinks rejection; pass that backend to the native wrapper. The failed constructor hunk moved from removed `get_kv_cache_layout()` to `get_flashinfer_layout_string(self.kv_cache_layout)`.

### 0106-dflash2-selector-sampling-v0280.diff

Verdict: **rebased; semantics-changed-needs-metal-check**.

Hunks: import Triton random primitives; remove the compile-time probabilistic flag; keep logits fp32; split deterministic and probabilistic selection, preserve fp64/fp32 noise arithmetic, and clamp the power-of-two sentinel; remove the obsolete launch argument.

Failed/moved hunk: upstream's selector now called `gumbel_noised_argmax(..., IS_DRAFTING=True)` and documented position keys. The local arithmetic replaces that new helper call, preserving upstream position semantics and adding the original clamp. Whether its distribution matches the target sampler still needs a metal comparison.

### 0107-dflash-quantized-draft-loader.diff

Verdict: **rebased; semantics-changed-needs-metal-check**.

Hunks: materialize a dense QKV weight through a quant layer's own kernel when `.weight` is absent; defer fused-KV construction when post-load repacking has not run. The failed loader hunk moved to rc1's `loader.load_weights(..., mapper=mapper)` path. The mapper and upstream draft `is_neox_style` layout are retained.

### 0108-gdn-kernels-v0280.diff

Verdict: **rebased; semantics-changed-needs-metal-check**.

Hunks: store active spec width; preallocate the arange; initialize width; recover runtime K; reuse persistent empty/token buffers; avoid zero-length copies; return width; use width for both GDN convolution call sites; bounds-check accepted-token state lookup in recurrent and sigmoid kernels.

Failed/moved hunk: rc1 adds an all-zero draft schedule reset. Runtime width is now computed immediately after the active mask, then rc1 is allowed to reset an empty schedule to ordinary decode. Geometry remains H=16, HV=48, K=V=128, convolution width 4.

### 0109-dflash-noncausal-complete-v0280.diff

Verdict: **rebased; semantics-changed-needs-metal-check**.

Hunks: allocate the non-causal wrapper pool; allocate persistent QO indptr; advertise non-causal uniform batches; construct graph-bound non-causal wrappers; key selection by request count. The failed buffer hunk moved beside rc1's `CpuGpuBuffer` page metadata, and the successful wrapper hunk was updated to the new layout conversion API.

### 0111-replayssm-spec-v0280.diff

Verdict: **rebased; semantics-changed-needs-metal-check**.

Hunks by file: `cache.py` adds the env/programmatic gate, hash behavior, and factors; `config/vllm.py` validates exclusivity, model/backend/cache mode, stochastic rounding, KV transfer, and ring length; `arg_utils.py` adds CLI/config plumbing; `mamba/abstract.py` selects the five-part ReplaySSM cache and recomputes padded bytes; `mamba_utils.py` defines five dtypes and shapes; `replayssm_config.py` supplies verify/flush/decode launch selectors; `qwen_gdn_linear_attn.py` selects those shapes/dtypes, records ring sizes, rejects the fused CUDA path, indexes recurrent dtype, bypasses ordinary rearrangement, invokes ReplaySSM verify, and binds the five states; `qwen3_5.py` exposes support and layers; `gdn_attn.py` adds persistent cursor state, lazy allocation, ReplaySSM prefill/decode classification, width-one state rows, prefill reset, accepted-token commit, first-decode reset, graph buffers, and returned cursor metadata; the new 650-line kernel implements reset, commit, verify, replay, and flush.

Failed/moved hunks: `cache.py` gained `functools.cache` and new retention/layout fields, so `os` and the gate moved beside the current ReplaySSM fields; the validator moved to the end of rc1's enlarged VllmConfig validator set; `MambaBase.get_kv_cache_spec` gained internal-checkpoint plumbing, which remains the off-path while the opt-in path creates its five states; Qwen init gained the upstream packed-decode capability branch, so local flags sit before its selector; GDN metadata now preserves rc1's zero-draft reset and uses `is_prefilling` only on the ReplaySSM branch. Open question: internal checkpoint retention plus ReplaySSM prefill resets is mechanically consistent but requires prefix-cache/preemption audition on metal.

### 0112-xqa-verify-v0280.diff

Verdict: **rebased; semantics-changed-needs-metal-check**.

Hunks: add `VLLM_SM12X_XQA_VERIFY`; derive verify eligibility; widen spec-as-decode; accept only exact uniform products; attach the packed causal mask; reshape generic XQA output for q_len>1; forward the mask.

Failed/moved hunks: rc1's dedicated XQA already supports ragged masks and `q_cu_seq_lens`; the local uniform NVFP4 branch is inserted before the new varlen trtllm-gen branch. The generic call retains rc1's `max_q_len`, cumulative lengths, and layout conversion while adding the mask.

### 0113-dflash-speculator-graphs-v0280.diff

Verdict: **verbatim**.

Hunks: import compilation types and expose FULL decode support to the non-causal DFlash2 attention builder while preserving runtime graph policy. No rc1 edit intersected the exact context.

### 0116-dflash-nvfp4-revival.diff

Verdict: **rebased; semantics-changed-needs-metal-check**.

Hunks: keep SM120 NVFP4 non-causal attention off graphs by default; reject SM100 causal-only trtllm-gen and select SM120 FA2; bypass the local graph pool unless explicitly re-admitted later; assert sinks stay non-NVFP4; mask invalid speculative ids at tp=1.

Failed/moved hunk: rc1 converted tp=1 embedding into an early return and removed the old shared mask tail. The mask/gather/zero sequence now lives wholly inside that fast branch; tp>1 rc1 fused and FP8 reduction paths are untouched.

### 0117-dflash-nvfp4-warmup.diff

Verdict: **verbatim**.

Hunks: make DFlash2 graph buffers persistent, preserve the warmup batch, initialize stable sampling state, and reuse it during warmup/capture. Exact rc1 context still applies.

### 0118b-dflash-eager-escape-rebased.diff

Verdict: **verbatim**.

Hunk: make `VLLM_SM12X_DFLASH_FORCE_EAGER=1` exit before draft capture. This remains an operator escape, not a root-cause claim.

### 0119-dflash-nvfp4-fullgraph-width.diff

Verdict: **verbatim**.

Hunk: calculate the NVFP4 prefill graph ceiling from the final uniform query width and request count rather than a stale maximum-spec assumption. It composes with rc1's new uniform capture list; the list controls reached sizes, while this field bounds the local pool.

### 0129-dflash-nvfp4-drafter-graphs.diff

Verdict: **verbatim; semantics-changed-needs-metal-check**.

Hunks: add/log `VLLM_SM12X_DFLASH_GRAPHS`; re-admit opted-in SM120 NVFP4 non-causal uniform graphs; cap the graph-bound DFlash wrapper pool and leave larger batches eager.

### 0131-nvfp4-pooled-int-workspace.diff

Verdict: **verbatim; semantics-changed-needs-metal-check**.

Hunks: add/log `_shrink_pooled_int_workspace` and `VLLM_SM12X_POOLED_INT_WS_MIB`; shrink causal pooled wrappers; shrink non-causal pooled wrappers. The patch's historical comment says the KV pool saw zero graph reserve; that premise is superseded by v0.29's pre-KV graph-memory profile, but the allocation reduction itself remains valid. Confirm FlashInfer planning does not exhaust 1 MiB.

## Phase 2

### 0132-masked-nvfp4-xqa-sm120-v0290.diff

Verdict: **rebased port; semantics-changed-needs-metal-check**.

Hunks: register `VLLM_FLASHINFER_XQA_USE_ISOLATED_STREAM`; resolve it from the environment; import context-manager types; lazily create a process-stable isolated stream and join/fork it capture-safely; admit NVFP4 to the dedicated XQA builder only when the existing verify gate is enabled; select the dedicated forward route only for masked NVFP4 batches; call dedicated XQA with packed K/V data, packed scale pages, model-dtype output, the mask, and optional isolated stream.

The model-dtype query rule was already absorbed by `0101`, and model-dtype output handling by `0103`, so 0132 does not duplicate them. PR #53543 directly changes the likely R155 failure surface: speculative verification no longer uses the older generic dispatcher/output reshape plus compact workspace scale copies; it uses FlashInfer's masked dedicated XQA API with the packed cache and scale views. It is a plausible fix, not local proof.

A/B cell: keep the known baseline `VLLM_SM12X_NVFP4_XQA=0` (FA2 verify). Candidate arm sets `VLLM_SM12X_NVFP4_XQA=1` and `VLLM_SM12X_XQA_VERIFY=1`; first run with `VLLM_FLASHINFER_XQA_USE_ISOLATED_STREAM=0`, then repeat with `=1` only to isolate stream ordering. Compare output fidelity/acceptance at identical prompts and dynamic-K tiers.

### 0133-gdn-packed-decode-bv16-v0290.diff

Verdict: **rebased port; semantics-changed-needs-metal-check**.

Hunks: register/parse `VLLM_SM12X_GDN_PACKED_BV`; add cached kernel launch selection; retain PR #54181's BV=16 default for H=HV=16, K=V=128, B<=24 on SM12x; let `16` or `32` force BV for any H/HV when K=V=128 on SM12x; use the selected BV/warps/stages in the wrapper. Benchmark/test files were intentionally omitted.

The PR's unforced selector never fires for this model because H=16 and HV=48. The env override is therefore the only local A/B path: compare unset/upstream selector, forced `16`, and forced `32` at B<=24 and around the B=24/25 boundary. Invalid forced values raise when the compatible SM12x K=V=128 path is reached.

## Risk on metal

1. Boot log first: require `linear-V-scale store overlay ACTIVE`, `use_fa2_nvfp4_kv` resolution, the chosen XQA fallback/route, `VLLM_SM12X_DFLASH_GRAPHS` status, and `SM12x pooled FlashInfer wrappers: int workspace shrunk 8 MiB -> ...`. Absence of the overlay line is a stop condition.
2. First functional probe: known-good FA2 cell (`VLLM_SM12X_NVFP4_XQA=0`) at q_len=1 and multi-token verification, including padded request batches and dynamic K=0. This catches layout, invalid-id mask, and uniform-width regressions.
3. 0132 probe: enable NVFP4 XQA plus verify, compare token/logit fidelity and acceptance against FA2, then toggle the isolated stream. A CUDA error, drift, or capture-only failure keeps 0132 unpromoted.
4. DFlash2 probe: verify ModelOpt/W4A16 load, upstream `is_neox_style`, non-causal FA2 selection, warmup completion, and eager/graph arms. Exercise the cap boundary and preemption/recompute.
5. ReplaySSM probe: opt in separately; cover prefill, first decode, K=0, rejection/acceptance, ring flush, prefix-cache hit, and preemption. Compare to ordinary GDN checkpoints.
6. 0133 probe: compare unset/BV16/BV32 for H=16, HV=48, K=V=128, checking exact output/state agreement before throughput or power.
7. Memory probe: compare v0.29 reported graph reserve and actual post-capture free memory with pooled workspace 1 versus 8 MiB. A planning workspace error means the 1 MiB default is too small for a reached shape.

## Verification transcript

Phase 1 was run and passed independently before phase 2. The final end-to-end rerun is reproduced exactly below.

```text
+ patch --dry-run -p1 --fuzz=0 -d tree/vllm < 0101-sm120-nvfp4kv-fa2-routing-v0280.diff
patching file 'v1/attention/backends/flashinfer.py'
patching file 'v1/worker/gpu_worker.py'
+ patch --dry-run -p1 --fuzz=0 -d tree/vllm < 0103-sm120-nvfp4-xqa-decode-v0280.diff
patching file 'v1/attention/backends/flashinfer.py'
+ patch --dry-run -p1 --fuzz=0 -d tree/vllm < 0104-mtp-drafter-full-cudagraph-v0280.diff
patching file 'v1/spec_decode/llm_base_proposer.py'
patching file 'v1/worker/gpu_model_runner.py'
+ patch --dry-run -p1 --fuzz=0 -d tree/vllm < 0105-dflash-noncausal-nvfp4-fa2-v0280.diff
patching file 'v1/attention/backends/flashinfer.py'
+ patch --dry-run -p1 --fuzz=0 -d tree/vllm < 0106-dflash2-selector-sampling-v0280.diff
patching file 'v1/worker/gpu/spec_decode/dflash2/speculator.py'
+ patch --dry-run -p1 --fuzz=0 -d tree/vllm < 0107-dflash-quantized-draft-loader.diff
patching file 'model_executor/models/qwen3_dflash.py'
+ patch --dry-run -p1 --fuzz=0 -d tree/vllm < 0108-gdn-kernels-v0280.diff
patching file 'v1/attention/backends/gdn_attn.py'
patching file 'model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py'
patching file 'third_party/flash_linear_attention/ops/fused_recurrent.py'
patching file 'third_party/flash_linear_attention/ops/fused_sigmoid_gating.py'
+ patch --dry-run -p1 --fuzz=0 -d tree/vllm < 0109-dflash-noncausal-complete-v0280.diff
patching file 'v1/attention/backends/flashinfer.py'
+ patch --dry-run -p1 --fuzz=0 -d tree/vllm < 0111-replayssm-spec-v0280.diff
patching file 'config/cache.py'
patching file 'config/vllm.py'
patching file 'engine/arg_utils.py'
patching file 'model_executor/layers/mamba/abstract.py'
patching file 'model_executor/layers/mamba/mamba_utils.py'
patching file 'model_executor/layers/mamba/ops/replayssm_config.py'
patching file 'model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py'
patching file 'model_executor/models/qwen3_5.py'
patching file 'v1/attention/backends/gdn_attn.py'
patching file 'model_executor/layers/fla/ops/gdn_replayssm_spec_decode.py'
+ patch --dry-run -p1 --fuzz=0 -d tree/vllm < 0112-xqa-verify-v0280.diff
patching file 'v1/attention/backends/flashinfer.py'
+ patch --dry-run -p1 --fuzz=0 -d tree/vllm < 0113-dflash-speculator-graphs-v0280.diff
patching file 'v1/worker/gpu/spec_decode/dflash2/speculator.py'
+ patch --dry-run -p1 --fuzz=0 -d tree < 0116-dflash-nvfp4-revival.diff
patching file 'vllm/v1/attention/backends/flashinfer.py'
patching file 'vllm/model_executor/layers/vocab_parallel_embedding.py'
+ patch --dry-run -p1 --fuzz=0 -d tree < 0117-dflash-nvfp4-warmup.diff
patching file 'vllm/model_executor/models/qwen3_dflash2.py'
patching file 'vllm/v1/worker/gpu/spec_decode/dflash/speculator.py'
+ patch --dry-run -p1 --fuzz=0 -d tree < 0118b-dflash-eager-escape-rebased.diff
patching file 'vllm/v1/worker/gpu/spec_decode/dflash/speculator.py'
+ patch --dry-run -p1 --fuzz=0 -d tree < 0119-dflash-nvfp4-fullgraph-width.diff
patching file 'vllm/v1/attention/backends/flashinfer.py'
+ patch --dry-run -p1 --fuzz=0 -d tree < 0129-dflash-nvfp4-drafter-graphs.diff
patching file 'vllm/v1/attention/backends/flashinfer.py'
+ patch --dry-run -p1 --fuzz=0 -d tree < 0131-nvfp4-pooled-int-workspace.diff
patching file 'vllm/v1/attention/backends/flashinfer.py'
+ patch --dry-run -p1 --fuzz=0 -d tree < 0132-masked-nvfp4-xqa-sm120-v0290.diff
patching file 'vllm/envs.py'
patching file 'vllm/v1/attention/backends/flashinfer.py'
+ patch --dry-run -p1 --fuzz=0 -d tree < 0133-gdn-packed-decode-bv16-v0290.diff
patching file 'vllm/envs.py'
patching file 'vllm/third_party/flash_linear_attention/ops/fused_recurrent.py'
+ python3 -m py_compile tree/vllm/config/cache.py
+ python3 -m py_compile tree/vllm/config/vllm.py
+ python3 -m py_compile tree/vllm/engine/arg_utils.py
+ python3 -m py_compile tree/vllm/envs.py
+ python3 -m py_compile tree/vllm/model_executor/layers/fla/ops/gdn_replayssm_spec_decode.py
+ python3 -m py_compile tree/vllm/model_executor/layers/mamba/abstract.py
+ python3 -m py_compile tree/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
+ python3 -m py_compile tree/vllm/model_executor/layers/mamba/mamba_utils.py
+ python3 -m py_compile tree/vllm/model_executor/layers/mamba/ops/replayssm_config.py
+ python3 -m py_compile tree/vllm/model_executor/layers/vocab_parallel_embedding.py
+ python3 -m py_compile tree/vllm/model_executor/models/qwen3_5.py
+ python3 -m py_compile tree/vllm/model_executor/models/qwen3_dflash.py
+ python3 -m py_compile tree/vllm/model_executor/models/qwen3_dflash2.py
+ python3 -m py_compile tree/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py
+ python3 -m py_compile tree/vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py
+ python3 -m py_compile tree/vllm/v1/attention/backends/flashinfer.py
+ python3 -m py_compile tree/vllm/v1/attention/backends/gdn_attn.py
+ python3 -m py_compile tree/vllm/v1/spec_decode/llm_base_proposer.py
+ python3 -m py_compile tree/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
+ python3 -m py_compile tree/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py
+ python3 -m py_compile tree/vllm/v1/worker/gpu_model_runner.py
+ python3 -m py_compile tree/vllm/v1/worker/gpu_worker.py
VERIFIED: phase 1 + 0132 + 0133
```
