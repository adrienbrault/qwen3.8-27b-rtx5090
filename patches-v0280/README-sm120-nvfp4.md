# sm120 NVFP4 KV cache rebase for vLLM v0.28.0

## Result

This package rebases the v0.27 SM12x NVFP4-KV route onto the supplied v0.28.0
sources. The primary patch sends only SM12x + NVFP4 through FlashInfer FA2,
keeps SM100/SM90 and SM12x fp8/auto routing unchanged, preserves FULL-CUDA-graph
speculative verification through an FA2 prefill wrapper, and sends NVFP4 stores
to the linear-V-scale overlay.

The V-scale verdict is **unchanged**: the stock v0.28.0 writer still swizzles V
scales unconditionally, so `0102` and the existing overlay are still required.
See `0102-vscale-writer-verdict-v0280.md` for the evidence audit.

## Artifacts

- `0101-sm120-nvfp4kv-fa2-routing-v0280.diff`: primary Python patch for
  `vllm/v1/attention/backends/flashinfer.py` and
  `vllm/v1/worker/gpu_worker.py`.
- `0102-nvfp4-writer-linear-vscale-sm12x-v0280.diff`: csrc writer patch against
  `csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`.
- `0102-vscale-writer-verdict-v0280.md`: explicit writer verdict and line-level
  evidence.
- `overlay/`: unchanged runtime-overlay builder, binding, loader, and numeric
  diagnostic copied from the validated v0.27 package.
- `0101a-ALTERNATIVE-dflash-noncausal-nvfp4-fa2-v0280.diff`: optional,
  apply-after-`0101` GPU A/B hunk that tries NVFP4 for DFlash non-causal
  attention. The supported primary fallback is per-layer skipping.

## Grounding in the supplied v0.28.0 sources

The new selector needs no patch. Its base class defines
`supports_kv_cache_dtype()` at `v0280-src/backend.py:167-172`, and
`validate_configuration()` calls the selected backend's override at
`v0280-src/backend.py:348-430` (specifically `:373-374`). The new
per-head-scale, connector, and adaptive-verification flags are already handled
by that validator at `v0280-src/backend.py:358-367` and `:393-416`; changing the
FlashInfer subclass gate is sufficient.

For those new flags, the rebase intentionally leaves the v0.28 contracts alone:
the base backend supports KV connectors at `v0280-src/backend.py:294-295`, while
FlashInfer opts out of device/CPU query-length mismatch at
`v0280-src/backends_flashinfer.py:449-453`, so the validator must continue to
reject adaptive verification for this backend. Per-head quant scales are not
part of this CLI-selected NVFP4 route, so `supports_per_head_quant_scales()` is
not widened from its base behavior at `v0280-src/backend.py:271-272`.

The stock FlashInfer NVFP4 gate accepts only SM100 with TRTLLM prefill and decode
at `v0280-src/backends_flashinfer.py:517-524`, while the supplied utility reports
that SM12x has XQA decode but no TRTLLM prefill at
`v0280-src/utils_flashinfer.py:395-417`. That combination explains the stock
SM120 rejection entirely from the staged sources.

The v0.28.0 cache helpers already implement the packed NVFP4 data/scale shape at
`v0280-src/torch_utils.py:531-559` and the random-cache layout at
`v0280-src/torch_utils.py:612-626`; they need no patch. The backend likewise
already gives NVFP4 separate K/V head slots at
`v0280-src/backends_flashinfer.py:391-405` and its four-dimensional cache shape
at `:467-479`.

## Per-hunk rationale for `0101`

| Diff hunk | Rationale and v0.28.0 evidence | Gate |
|---|---|---|
| `flashinfer.py` 4 | Import `cache` for the one-time lazy overlay resolver added by hunk 2660. | Resolver is called only by the FA2 NVFP4 store route. |
| 516 | Admit NVFP4 on SM12x before retaining the exact stock SM100+TRTLLM predicate. Selector delegation is `v0280-src/backend.py:373-374`. | dtype starts with `nvfp4`; family 120. |
| 564 | Require HND for the global SM12x NVFP4 request; stock HND-on-SM100 and SM12x fp8/auto remain unchanged. | major 12; configured dtype starts with `nvfp4`. |
| 787 | Set the per-group FA2 route flag and bypass the SM100 TRTLLM requirement only on SM12x. Both scale-search variants still normalize the reader dtype to `nvfp4`. | per-group NVFP4; family 120. |
| 808 | Initialize the false route for unquantized skip groups, then add graph-prefill state only for FULL-graph FA2 NVFP4. The uniform width mirrors v0.28 parallel-drafting semantics. | `enable_cuda_graph and use_fa2_nvfp4_kv`. |
| 836 | Clear XQA/trtllm-gen decode before `use_dedicated_xqa` and reorder-threshold calculation. This sends q_len=1 to FA2 decode and q_len>1 verification to FA2 prefill. | `use_fa2_nvfp4_kv`. |
| 924 | Allocate the persistent GPU query-indptr buffer needed by the graph-bound prefill wrapper. | FULL graph plus `use_fa2_nvfp4_kv`. |
| 960 | Use BF16/FP16 model-dtype Q for FA2; retain FP8-Q for SM100 trtllm-gen. | inside the NVFP4 branch; family 120. |
| 1119 | Add optional batch/graph arguments to `_get_prefill_wrapper`; defaults preserve every existing caller. | Graph values are supplied only by hunk 1568. |
| 1154 | Create fixed-address paged-prefill wrappers for uniform verification capture. The branch asserts the FA2 NVFP4 route. | `use_cudagraph`, reachable only from the gated caller. |
| 1174 | Select wrapper backend `auto` (FA2 on this stack) for SM12x NVFP4; keep `trtllm-gen` for SM100 NVFP4 and stock `auto` otherwise. | `use_fa2_nvfp4_kv`. |
| 1200 | Select the explicit `fa2` decode wrapper only for SM12x NVFP4; retain all other decode backends. | `is_kvcache_nvfp4` plus route flag. |
| 1517 | Pass uint8 KV metadata to the native cascade planner. Cascade is already disabled for quantized KV, but the argument is kept internally consistent. | `use_fa2_nvfp4_kv`. |
| 1568 | Route only pure, uniform, in-range speculative batches through graph prefill. DCP is excluded because its wrapper has different KV plumbing. | FA2 NVFP4, FULL graph, no DCP, exact uniform width. |
| 1594 | Pass uint8 to `BatchDCPPrefillWrapper.plan` for native-reader consistency. DCP forward remains unsupported and is called out under Risks. | `use_fa2_nvfp4_kv`. |
| 1603 | Plan native prefill output as model dtype on FA2 while retaining FP8 output on SM100 trtllm-gen. | `use_fa2_nvfp4_kv`. |
| 1623 | Pass uint8 KV data to the native FA2 prefill plan rather than the TRTLLM-only NVFP4 string. | `use_fa2_nvfp4_kv`. |
| 1688 | Plan native decode output as model dtype on FA2 while retaining SM100's FP8 intermediate contract. | `use_fa2_nvfp4_kv`. |
| 1711 | Pass uint8 KV data to native FA2 decode. | `use_fa2_nvfp4_kv`. |
| 1762 | Mirror the route flag in `FlashInferImpl`, including support for both `nvfp4` and `nvfp4_4over6`. | dtype starts with `nvfp4`; family 120. |
| 1808 | Do not allocate the SM100-only FP8 output scratch buffer when FA2 writes model dtype directly. | NVFP4 and not `use_fa2_nvfp4_kv`. |
| 2124 | Native FA2 prefill writes directly to BF16/FP16 output; retain conversion for SM100. | `use_fa2_nvfp4_kv`. |
| 2296 | Native FA2 decode likewise writes directly to BF16/FP16 output. | `use_fa2_nvfp4_kv`. |
| 2529 | Route stores through the cached overlay writer and add loud diagnostic fallback logging. The original dtype string is forwarded, preserving `nvfp4_4over6`. | `use_fa2_nvfp4_kv`. |
| `gpu_worker.py` 510 | Run one warmup before the memory window so FlashInfer JIT code does not reduce the measured KV pool. Stock paths still execute only their original profiled run. | family 120; configured dtype starts with `nvfp4`. |

No `backend.py` hunk is needed for the selector reason above. No
`utils_flashinfer.py` hunk is needed because its v0.28 implementation already
describes SM12x as XQA-decode-only (`v0280-src/utils_flashinfer.py:399-417`);
`0101` instead disables that decode route only for NVFP4. No `cache.py` or
`torch_utils.py` change is needed.

## Per-hunk rationale for `0102` and writer verdict

The stock writer defines the four-token SM100 scale swizzle at
`refs/nvfp4_kv_cache_kernels-v0280.cu:34-48`. Its write branch is still K-linear
at `:282-290` and unconditionally V-swizzled at `:291-299`. There is no upstream
linear/device path: the swizzle-specific block-size check remains unconditional
at `:333-337`, and both `nvfp4` and `nvfp4_4over6` call the same layout-unaware
kernel signature at `:388-417`. The `4over6` enum is only a scale-search choice
at `:28-32`.

`0102` therefore applies the v0.27 writer fix unchanged:

1. Add a `swizzle_v_sf` kernel argument.
2. Write V scales through the same linear address as K when the flag is false.
3. Set the flag from `get_device_prop()->major < 12`, preserving the SM100
   swizzle and making the divisibility check conditional on it.
4. Pass the flag through the `nvfp4` dispatch.
5. Pass the same flag through `nvfp4_4over6`; that variant changes scale search,
   not layout, so it needs the identical reader-layout choice.

The unchanged overlay binding remains compatible with the writer's dispatch
entry point at `refs/nvfp4_kv_cache_kernels-v0280.cu:307-315`. The copies under
`out/overlay/` compare byte-for-byte equal to `patches-nvfp4kv/overlay/`.

## DFlash primary fallback and clearly marked ALTERNATIVE

The primary `0101` deliberately retains v0.28's conservative rejection of
non-causal NVFP4 at `v0280-src/backends_flashinfer.py:1128-1132`. For the first
DFlash boot, configure `--kv-cache-dtype-skip-layers` to match the draft
attention layers, leaving target-model attention NVFP4. v0.28 accepts layer
indices or attention-type patterns at `v0280-src/cache.py:114-116` and carries a
separate padded page size for skipped layers at `:120-124`.

**ALTERNATIVE — GPU A/B only:** after `0101`, apply
`0101a-ALTERNATIVE-dflash-noncausal-nvfp4-fa2-v0280.diff` to relax that one guard
only when `use_fa2_nvfp4_kv` is true. The remaining plan and forward path already
use BF16/FP16 Q/O, uint8 KV data, and separate block scales, so this is the
smallest reasonable experiment; FlashInfer 0.6.16.post3 non-causal NVFP4 support
cannot be proven from the supplied files. Do not promote it on boot alone: A/B
fidelity and deep-context decode, and revert to the skip-layer primary on any
failure.

## Exact apply/build instructions

The commands below assume the `out/` directory is mounted at `/work/out` in the
image/build stage. The Python diff uses the same `a/v1/...` paths as the older
overlay and applies from the installed `vllm` package root.

```bash
VROOT=/usr/local/lib/python3.12/dist-packages/vllm

patch --batch --forward --dry-run -p1 -d "$VROOT" \
  < /work/out/0101-sm120-nvfp4kv-fa2-routing-v0280.diff
patch --batch --forward -p1 -d "$VROOT" \
  < /work/out/0101-sm120-nvfp4kv-fa2-routing-v0280.diff

python3 -m py_compile \
  "$VROOT/v1/attention/backends/flashinfer.py" \
  "$VROOT/v1/worker/gpu_worker.py"
```

For the DFlash ALTERNATIVE only, after applying `0101`:

```bash
VROOT=/usr/local/lib/python3.12/dist-packages/vllm
patch --batch --forward --dry-run -p1 -d "$VROOT" \
  < /work/out/0101a-ALTERNATIVE-dflash-noncausal-nvfp4-fa2-v0280.diff
patch --batch --forward -p1 -d "$VROOT" \
  < /work/out/0101a-ALTERNATIVE-dflash-noncausal-nvfp4-fa2-v0280.diff
```

For the runtime overlay, stage the complete v0.28.0 vLLM csrc tree (headers
included) at `/opt/vllm-src`; its writer must match the supplied reference.
Then patch and build:

```bash
CSRC_ROOT=/opt/vllm-src

patch --batch --forward --dry-run -p1 -d "$CSRC_ROOT" \
  < /work/out/0102-nvfp4-writer-linear-vscale-sm12x-v0280.diff
patch --batch --forward -p1 -d "$CSRC_ROOT" \
  < /work/out/0102-nvfp4-writer-linear-vscale-sm12x-v0280.diff

install -d /opt/vllm-sm12x
cp -a /work/out/overlay/. /opt/vllm-sm12x/

export VLLM_SM12X_CSRC=/opt/vllm-src/csrc
export VLLM_SM12X_BUILD=/opt/vllm-sm12x/build
export VLLM_SM12X_ARCH=120a
export PYTHONPATH="/opt/vllm-sm12x${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_CUDA_ARCH_LIST=12.0a
export MAX_JOBS=4

python3 /opt/vllm-sm12x/build_overlay.py
python3 -c "import torch; torch.ops.load_library('/opt/vllm-sm12x/build/vllm_sm12x_nvfp4kv.so'); assert hasattr(torch.ops.vllm_sm12x, 'reshape_and_cache_nvfp4')"
```

Persist `VLLM_SM12X_CSRC`, `VLLM_SM12X_BUILD`, `VLLM_SM12X_ARCH`, and
`PYTHONPATH` as Docker `ENV` values, as in the existing overlay build. Retain
the existing torch stable-string shadow-header handling if its probe fires;
`build_overlay.py` already accepts `VLLM_SM12X_SHADOW_INC`.

At service startup, treat absence of both the log line
`NVFP4KV-SM120: linear-V-scale store overlay ACTIVE` and the compiled `.so` as
fatal. The Python route deliberately has a diagnostic fallback, but the stock
writer is not quality-correct for this FA2 route.

## v0.27 piece disposition

| v0.27 artifact | v0.28 disposition |
|---|---|
| `0001-sm120-nvfp4kv-fa2-routing.diff` | Superseded by `0101`; surrounding routing, XQA, metadata, and output handling changed in v0.28, so do not apply the old patch. |
| `0001b-prefill-wrapper-signature.diff` | Obsolete as a standalone patch; its two parameters are folded into the rebased `0101` wrapper hunk. |
| `0002-nvfp4-writer-linear-vscale-sm12x.diff` | Logic is unchanged, but use the audited v0.28-named `0102` copy. It dry-runs cleanly against the supplied v0.28 writer. |
| `0002b-flashinfer-route-nvfp4-store-to-overlay.diff` | Obsolete as a standalone patch; rebased and folded into `0101` at the v0.28 `do_kv_cache_update()` location (`v0280-src/backends_flashinfer.py:2504-2541`). |
| `overlay/` | Not obsolete and not modified; copied into `out/overlay/`. |
| `0003-pr49891-mtp-drafter-full-cudagraph.diff` | Not included. It is an optional drafter performance lever, not required for the NVFP4 correctness route. The files it patches are not in the supplied v0.28 source subset, so no claim is made here about whether v0.28 integrated or still needs it. |
| `upstream-pr49891-original.diff` | Audit reference only; do not apply it on v0.28. |
| `Dockerfile.nvfp4kv` | Obsolete for v0.28: it pins the v0.27 image digest, commit `ac7509e2b`, version check, and old patch names. Recreate its overlay-build steps against the v0.28.0 image/csrc using the commands above. |

## Validation checklist

- [x] `0101` dry-runs with `patch --dry-run -p1` against clean copies of the
  supplied `backends_flashinfer.py` and `gpu_worker.py`.
- [x] The DFlash ALTERNATIVE dry-runs after `0101`, applies exactly, and the
  resulting backend passes `python3 -m py_compile`.
- [x] `0102` dry-runs with `patch --dry-run -p1` against
  `refs/nvfp4_kv_cache_kernels-v0280.cu` staged at its csrc path.
- [x] Applying both diffs to scratch copies reproduces the reviewed patched
  files exactly.
- [x] Patched Python files pass `python3 -m py_compile`.
- [x] All copied overlay files are byte-for-byte equal to the v0.27 overlay.
- [ ] Build the overlay for `sm_120a`; verify the shared object registers
  `torch.ops.vllm_sm12x.reshape_and_cache_nvfp4`.
- [ ] Boot with async scheduling left at its v0.28 default. Record
  `kv_cache_dtype=nvfp4`, `FLASHINFER`, `decode_backend=flashinfer-native`, HND
  layout, the overlay ACTIVE line, block size, pool size, maximum concurrency,
  and MTP metrics. Expected pool is approximately 330-400K tokens at utilization
  0.93.
- [ ] Run an fp8/auto control on the patched image and compare it with stock;
  routing and outputs must be unchanged. Every behavioral `0101` change is
  gated on SM12x NVFP4, including the extra JIT warmup.
- [ ] Run the fidelity ruler against the validated FP8 reference. This is the
  promotion blocker for the V-scale layout; expected rough band is top-1 about
  0.89 and delta-NLL about 2.1%, within noise of the prior stack.
- [ ] Run the writer diagnostic A/B: overlay enabled versus
  `VLLM_SM12X_NVFP4_LINEAR_VSF=0`. The swizzled diagnostic must materially lose
  fidelity/deep recall. Never serve with the variable set to `0`.
- [ ] Run cold and warm depth needles at 9K, 20K, 40K, 60K, then 100K/150K as
  available, both MTP off and `qwen3_5_mtp` with `ns=4`.
- [ ] Record MTP acceptance and verify uniform `1+ns` batches use native FA2
  prefill without disabling FULL CUDA graphs or async scheduling.
- [ ] Keep adaptive verification disabled for this route; v0.28 FlashInfer
  explicitly opts out of the required device/CPU query-length mismatch. This is
  independent of async scheduling, which should remain enabled.
- [ ] Boot the DFlash draft with its attention layers covered by
  `--kv-cache-dtype-skip-layers`; confirm target attention remains NVFP4.
  Optionally A/B the clearly marked non-causal NVFP4 alternative, then repeat
  fidelity and deep-context decode before considering it.
- [ ] Run deep-context decode performance and quality versus fp8, including the
  target concurrency/load shape; do not infer deep performance from short
  prompts.
- [ ] Exercise repeated/cached multi-request recall under load. Treat fluent but
  wrong retrieval as a hard failure even when there is no crash or NaN.

## Risks and uncompleted checks

- No GPU is available here, so FlashInfer JIT selection, CUDA-graph capture,
  numeric fidelity, MTP acceptance, DFlash behavior, and pool size remain to be
  validated on the RTX 5090.
- Assumption: this CLI-only NVFP4 target presents
  `use_per_head_quant_scales=False` to the selector. If the serving config sets
  it true, v0.28's base capability remains false at
  `v0280-src/backend.py:271-272` and selection will reject FlashInfer; this
  rebase does not claim support for that separate scale mode.
- The staged work package has the v0.28 writer file but not the complete csrc
  include tree or `nvcc` build context, so the overlay could not be compiled
  locally. Its binding signature was checked against
  `refs/nvfp4_kv_cache_kernels-v0280.cu:311-315`, and its source is unchanged.
- SM12x NVFP4 with DCP is outside this single-GPU target and is not supported by
  this patch: the v0.28 `BatchDCPPrefillWrapper.run()` path has no
  `kv_cache_sf` argument when it invokes the context wrapper at
  `v0280-src/backends_flashinfer.py:346-364`. The graph-prefill route is
  explicitly disabled when DCP is active.
- The optional DFlash non-causal relaxation is explicitly an on-GPU A/B item. A
  complete, apply-ready alternative is supplied; the primary uses v0.28's
  supported per-layer skip fallback.
- A missing overlay falls back with a warning to aid the diagnostic switch, but
  that fallback is known-invalid for serving because the v0.28 writer remains
  V-swizzled (`refs/nvfp4_kv_cache_kernels-v0280.cu:282-299`). Make the launcher
  fail closed if the ACTIVE line or `.so` is absent.

No other BRIEF requirement was skipped. The only incomplete work is the
GPU/nvcc-dependent validation explicitly listed above.
