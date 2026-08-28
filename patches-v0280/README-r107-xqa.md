# R107: SM120 NVFP4 XQA decode and drafter FULL CUDA graphs

## Result

This package closes the R106 single-token decode gap without changing the
validated NVFP4 writer or prefill path:

- SM120 + NVFP4 + `VLLM_SM12X_NVFP4_XQA!=0` routes only `q_len=1` decode to
  FlashInfer's XQA-NVFP4 kernel.
- Prefill and speculative verification (`q_len>1`) remain on the R106
  FlashInfer FA2 path, including its graph-bound prefill wrappers.
- `VLLM_SM12X_NVFP4_XQA=0` restores the current R106 FA2 decode route at
  builder initialization, before CUDA-graph capture.
- FP8/auto, SM100, and SM90 keep their existing routes. The decode path is
  gated by the existing SM120-NVFP4 flag; the drafter rebase independently
  rechecks family 120 and an `nvfp4*` cache dtype before enabling FULL mode.
- v0.28.0 has **not** absorbed the old PR #49891 drafter FULL-cudagraph work.
  `0104` is a rebased and narrowed two-file patch that enables FULL mode only
  for compatible LLM-based proposers; DFlash and extract-hidden-states keep
  their v0.28 behavior.

The frozen linear-V-scale overlay from R106/`0102` remains mandatory. Neither
artifact here changes its writer layout or store operation.

## Artifacts

| Artifact | Purpose |
|---|---|
| `0103-sm120-nvfp4-xqa-decode-v0280.diff` | Apply to the **R106-patched** `v1/attention/backends/flashinfer.py`; sends single-token SM120 NVFP4 decode to XQA and supplies compact scale-factor views. |
| `0104-mtp-drafter-full-cudagraph-v0280.diff` | Rebase of the substantive old `0003` behavior onto the supplied v0.28.0 `llm_base_proposer.py` and `gpu_model_runner.py`, narrowed to supported SM120 NVFP4 proposers. |
| `README.md` | Layout proof, routing rationale, exact application steps, risks, and validation checklist. |

SHA-256 values from the verified artifacts:

```text
88309b0460fa2a907423b53ad00dbc4f846e1c8b0a410dc458aee11f9b700c2d  0103-sm120-nvfp4-xqa-decode-v0280.diff
23c1308b5ea11d2d5db175bdb95e10cdf57a1c508bbea68988e1062190bbdeab  0104-mtp-drafter-full-cudagraph-v0280.diff
```

## Decode routing after `0103`

| Device/cache/batch | Route |
|---|---|
| SM120, NVFP4, env unset/nonzero, causal `q_len=1` | XQA-NVFP4 through FlashInfer's generic XQA decode API |
| SM120, NVFP4, env unset/nonzero, `q_len>1` | Existing FA2 prefill/spec-verification route |
| SM120, NVFP4, `VLLM_SM12X_NVFP4_XQA=0` | Existing FA2 decode/prefill route |
| SM120 FP8/auto | Stock v0.28 XQA/native behavior, unchanged |
| SM100/SM90 or non-NVFP4 | Stock behavior, unchanged |

The backend keeps `use_fa2_nvfp4_kv=True`; that flag still owns prefill,
speculative verification, and the linear writer. A new
`use_xqa_nvfp4_decode` flag merely preserves the already selected XQA kernel
for supported single-token decode. NVFP4 is deliberately excluded from
`use_dedicated_xqa`, so `_init_reorder_batch_threshold()` receives
`supports_spec_as_decode=False`. The threshold therefore stays at 1:
single-token requests enter decode and wider verification requests enter
prefill.

The generic call is intentional. It reaches
`trtllm_batch_decode_with_kv_cache(..., backend="xqa")`, which validates the
E4M3 scale tensors and then calls `xqa_batch_decode_with_kv_cache`. The
dedicated SM12x speculative/ragged call remains FP8-only and byte-equivalent
for its existing users.

## Packed-cache layout and stride arithmetic

Let:

```text
P = number of pages visible to one layer
H = KV heads per rank
T = tokens per page
D = logical head dimension
U = D / 2       packed FP4 data bytes per head/token
S = D / 16      E4M3 scale bytes per head/token
F = U + S       = 9D / 16 bytes
sigma = kv_cache.stride(0), in uint8 elements
```

R106 forces HND only for SM120 NVFP4. The logical allocation is
`[P, 2H, T, F]`, with K heads followed by V heads
(`backends_flashinfer_PATCHED.py:468-479,567-582,2164-2178`). The writer states
the physical regions explicitly as
`[K_data | K_scale | V_data | V_scale]`
(`refs/nvfp4_kv_cache_kernels-v0280.cu:8-10,353-359`).

Within page `p`, for head `h`, token `t`, packed byte `j`, and scale group `s`:

```text
page_base = p * sigma

K_data[p,h,t,j] = page_base + (h*T + t)*U + j
K_sf  [p,h,t,s] = page_base + H*T*U + (h*T + t)*S + s
V_data[p,h,t,j] = page_base + H*T*F + (h*T + t)*U + j
V_sf  [p,h,t,s] = page_base + H*T*F + H*T*U + (h*T + t)*S + s
```

`j` stores channels `2j` and `2j+1`; scale byte `s` applies to channels
`16s..16s+15`. The writer emits eight packed FP4 bytes plus one scale per 16
logical values (`refs/nvfp4_kv_cache_kernels-v0280.cu:249-299`).

For this model's `D=256`:

```text
U = 128 bytes
S =  16 bytes
F = 144 bytes

packed per-page size = 288*H*T bytes
K_data offset =   0
K_sf   offset = 128*H*T
V_data offset = 144*H*T
V_sf   offset = 272*H*T
end             = 288*H*T
```

`nvfp4_split_data_scale()` reconstructs zero-copy HND views from these regions
(`v0280-src/torch_utils.py:531-583`):

```text
K/V data: shape [P,H,T,U], strides [sigma,T*U,U,1]
K/V SF:   shape [P,H,T,S], strides [sigma,T*S,S,1]
```

For a standalone contiguous layer, `sigma=2*H*T*F`. The data page stride is
therefore `2F/U = 9/4` times its compact stride, while the SF page stride is
`2F/S = 18` times compact. v0.28 can pack layers into a shared backing tensor,
making `sigma` larger still; the helper correctly preserves that actual page
stride.

XQA documents packed data as `[P,H,T,D/2]` and scales as
`[P,H,T,D/16]` for HND (`fi_xqa.py:222-237`). Its HND normalization is a
transpose-only conversion to NHD (`fi_xqa.py:377-385`):

```text
data: [P,T,H,U], strides [sigma,U,T*U,1]
SF:   [P,T,H,S], strides [sigma,S,T*S,1]
```

### Stride verdict

The primary patch keeps K/V data zero-copy but compacts both SF tensors.

Zero-copy K/V is supported by evidence in the supplied stack: the working FP8
XQA route already consumes K/V views split from packed storage, XQA's HND path
is explicitly transpose-only, and the JIT key specializes dtype/page/head
geometry rather than strides (`fi_jit_xqa.py:53-165`). R106 also validated the
same NVFP4 data views through FA2.

SF is different: it is a new tensor input with the unusual 18x-or-larger page
stride, the lower generated XQA C++ is not supplied, and issue #49011 observed
packed strided views being misread. `0103` therefore copies K and V scales to
compact HND views:

```text
shape   [P,H,T,S]
strides [H*T*S,T*S,S,1]
```

After FlashInfer's transpose, each becomes the expected NHD view with strides
`[H*T*S,S,T*S,1]`.

Two mirrors require:

```text
2*P*H*T*S = P*H*T*D/8 bytes
```

That is exactly one ninth of one layer's packed NVFP4 cache. No separate CUDA
allocation is made for the mirrors: they are views carved from the tail of the
persistent TRTLLM/XQA workspace. Each `FlashInferImpl` caches only view
metadata, rebuilding it when startup moves from the minimal profiling cache to
the final cache shape. Every step records two fixed-address `copy_` operations;
there is no `.contiguous()` allocation in the decode loop.

The non-overlapping workspace prefix is passed to XQA. FlashInfer reserves its
first 8 MiB for semaphores (`fi_decode.py:3670-3672`), so the code fails early
if:

```text
workspace_bytes <= 8 MiB + 2*P*H*T*S
```

Increase `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE` or set
`VLLM_SM12X_NVFP4_XQA=0` if that guard fires. The configured prefix may need
more than the known 8 MiB minimum for the kernel's own scratch; this is an
on-GPU boot check.

The compact scales stay `torch.float8_e4m3fn`. Although the lower XQA docstring
calls them uint8 bytes, the high-level API used here explicitly requires E4M3
(`fi_decode.py:3242-3257`); both representations occupy one byte.

## `0103` per-hunk rationale

| Base location | Change and reason | Gate |
|---|---|---|
| 106 | Cached `VLLM_SM12X_NVFP4_XQA` switch; literal `0` logs the FA2 fallback. | Called only while constructing an SM120 NVFP4 route. |
| 867 | Compute `use_xqa_nvfp4_decode` from existing FA2-NVFP4 state, XQA head support, no DCP, and XQA's `D<=256`, `D%16==0` limits. | SM120 NVFP4 through `use_fa2_nvfp4_kv`. |
| 881 | Clear XQA only for env-off or unsupported NVFP4; preserve the selected SM120 XQA kernel otherwise. | SM120 NVFP4 only. |
| 901 | Exclude NVFP4 from the dedicated speculative/ragged XQA flag so the reorder threshold remains 1. | `not is_kvcache_nvfp4`; FP8 remains equivalent. |
| 1941 | Cache persistent workspace/SF view metadata without allocating device memory. | `use_fa2_nvfp4_kv` only. |
| 2005 | Validate E4M3 SF geometry, reserve a non-overlapping XQA prefix, create compact views, and copy the strided SF inputs. | Called only by NVFP4 XQA. |
| 2181 | Keep the direct dedicated XQA forward branch FP8-only; NVFP4 falls through to the generic XQA API that accepts SF tensors. | XQA plus NVFP4 test. |
| 2487 | Substitute compact SF scratch before the generic call; block tables and sequence lengths stay on the existing persistent GPU buffers. | XQA plus NVFP4. |
| 2556 | Allocate/use the FP8 output intermediate only for SM100 trtllm-gen. SM120 XQA writes BF16/FP16 directly. | NVFP4 plus trtllm-gen. |
| 2614 | Pass compact scale tensors with the zero-copy packed K/V data. | NVFP4 only. |

## CUDA-graph analysis

The q_len=1 XQA call mirrors v0.28's working FP8 XQA graph route:

- FlashInfer registers XQA as a custom op with output/workspace mutations and a
  fake implementation (`fi_xqa.py:94-180`).
- The cached JIT module key includes dtype, page size, head dimension, group
  ratio, sliding-window state, and query width—not batch size
  (`fi_xqa.py:67-92,404-455`; `fi_jit_xqa.py:53-165`).
- NVFP4 is explicitly accepted only on SM120 (`fi_xqa.py:394-400`). The target
  page sizes and `D=256` satisfy the JIT limits (`fi_jit_xqa.py:82-91`).
- Block tables, sequence lengths, output, the global workspace, and compact SF
  views all have stable storage. With the target FULL configuration, v0.28's
  CUDA-graph memory profiler performs warmup/capture against a minimal cache;
  the final-cache shape then rebuilds only view metadata before live capture.
- The SF `copy_` calls are ordinary fixed-address CUDA operations and become
  part of every captured graph. Capture batch sizes from 1 through 80 do not
  change the full cache shape or JIT key.
- q_len>1 never reaches this XQA route. It retains R106's persistent FA2
  prefill wrappers for FULL-graph speculative verification.

## `0104`: v0.28 drafter verdict and rebase

The old `0003` is still applicable; v0.28 did not absorb equivalent behavior.
Evidence in the supplied unpatched files:

- `llm_base_proposer.py:163` still describes a PIECEWISE-only dispatcher.
- `llm_base_proposer.py:419-434` maps target FULL/mixed mode to PIECEWISE.
- First and later draft passes omit uniform-decode dispatch at `:569-570` and
  `:667-668`.
- Capture-time `dummy_run()` has no target-mode input at `:1637-1652`.
- `_determine_batch_execution_and_padding()` does not accept forced mode or
  uniform decode at `:1803-1812`.
- `gpu_model_runner.py:6293-6305` explicitly permits drafter capture only when
  the target mode is PIECEWISE, and its call at `:6317-6322` forwards no target
  mode.

`0104` ports the substantive PR behavior with three safety narrowings:

1. A proposer-local policy flag starts false. It becomes true only when the
   runner opts in, the device is family 120, the global cache dtype starts with
   `nvfp4`, eager enforcement is off, and the resolved target mode supports
   PIECEWISE/FULL dispatch.
2. The runner opts in only `EagleProposer`, `DraftModelProposer`, and
   `Gemma4Proposer`. DFlash and extract-hidden-states receive the exact original
   initialization and `dummy_run()` calls, without new keyword arguments.
3. Uniform-decode hints, target-mode forcing, and the DP uniform redispatch are
   all conditional on that proposer-local flag. Non-SM120, non-NVFP4, FP8/auto,
   DFlash, and extract paths execute the stock PIECEWISE branches.

`target_cudagraph_mode` is passed only during compatible FULL capture and only
while `use_cudagraphs` remains true. Enforce-eager and the existing LoRA
graph-disable condition therefore remain authoritative. The old signature-only
DFlash/extract hunks are unnecessary and are deliberately absent.

Per-hunk behavior:

| File/location | Rationale |
|---|---|
| `llm_base_proposer.py:160` | Initialize the persistent FULL-policy flag false. |
| `:419` | Add `allow_full` and recheck SM120, NVFP4, eager, and resolved-mode gates before using target FULL keys. |
| `:566` | On the gated path, mark the first pass uniform only when `max_query_len==1`; otherwise make the original dispatch call. |
| `:665` | Mark later autoregressive draft passes uniform only on the gated path. |
| `:1637` | Accept and honor the target capture mode only during gated graph capture. |
| `:1804` | Conditionally forward uniform state, build the forced FULL descriptor, and preserve uniform classification during DP redispatch; all fallback calls match stock. |
| `gpu_model_runner.py:6290` | Permit FULL capture only behind the platform + cache dtype + proposer-type gate while preserving eager/LoRA disables. |
| `:6314` | Forward the target FULL mode only to the compatible `dummy_run()` signature. |
| `:7350` | Pass `allow_full=True` only under the same triple gate; retain the original initializer call for DFlash/extract and all non-targets. |

The rebase retains the upstream use of the dispatcher's private
`_create_padded_batch_descriptor()` method and its `has_lora=False` capture
assumption. The specialized-LoRA disable prevents the known conflicting case,
but other LoRA and multi-GPU DP combinations remain runtime risks; this work
package targets one RTX 5090 without LoRA.

## Exact apply order and commands

`0103` is **not** a stock-v0.28 diff. It is against the supplied
`r107-src/backends_flashinfer_PATCHED.py`, which is the installed file after
R106 `0101`. Apply in this order:

1. Start from the validated R106 image with `0101` and frozen `0102`/overlay
   already present. Do not reapply them to that image.
2. Apply `0103` at the installed vLLM package root.
3. Apply `0104` at the same root. It touches independent files and is the
   drafter performance part of R107.

Assuming `out-r107/` is mounted at `/work/out-r107`:

```bash
VROOT=/usr/local/lib/python3.12/dist-packages/vllm

patch --batch --forward --dry-run -p1 -d "$VROOT" \
  < /work/out-r107/0103-sm120-nvfp4-xqa-decode-v0280.diff
patch --batch --forward -p1 -d "$VROOT" \
  < /work/out-r107/0103-sm120-nvfp4-xqa-decode-v0280.diff

python3 -m py_compile \
  "$VROOT/v1/attention/backends/flashinfer.py"

patch --batch --forward --dry-run -p1 -d "$VROOT" \
  < /work/out-r107/0104-mtp-drafter-full-cudagraph-v0280.diff
patch --batch --forward -p1 -d "$VROOT" \
  < /work/out-r107/0104-mtp-drafter-full-cudagraph-v0280.diff

python3 -m py_compile \
  "$VROOT/v1/spec_decode/llm_base_proposer.py" \
  "$VROOT/v1/worker/gpu_model_runner.py"
```

Do not apply `patches-nvfp4kv/0003-pr49891-mtp-drafter-full-cudagraph.diff`
on v0.28. Use `0104`; it is rebased and avoids requiring unprovided
`dflash.py`/`extract_hidden_states.py` signature changes.

The environment switch is resolved while metadata builders are initialized,
and the decode kernel is static for FULL graphs. Restart the engine between
XQA and FA2 A/B runs:

```bash
# Primary
unset VLLM_SM12X_NVFP4_XQA

# R106 decode fallback / A-B control
export VLLM_SM12X_NVFP4_XQA=0
```

Leave async scheduling at the v0.28 default in both runs.

## Local verification completed

- [x] `0103` dry-runs and applies with `patch --dry-run -p1`/`patch -p1`
  against a scratch tree containing the exact supplied
  `backends_flashinfer_PATCHED.py` at
  `v1/attention/backends/flashinfer.py`.
- [x] Its applied output is byte-for-byte equal to the reviewed R107 candidate.
- [x] The patched backend passes `python3 -m py_compile`.
- [x] `0104` dry-runs and applies against scratch copies of the exact supplied
  v0.28 `llm_base_proposer.py` and `gpu_model_runner.py` at their `a/v1/...`
  paths.
- [x] Both `0104` outputs pass `python3 -m py_compile`.
- [x] No source or overlay file outside the three Python targets is patched.

## Required on-GPU validation

- [ ] Boot with the overlay ACTIVE and XQA enabled. Assert the backend log says
  `decode_backend=xqa`, `kv_cache_dtype=nvfp4`, and `arch=sm120`. Treat
  workspace-size errors, JIT fallback, a missing overlay, or graph-capture
  failure as a hard failure. Record pool size and maximum concurrency. Unlike
  R106 native decode, XQA allocates the global TRTLLM workspace, so the pool may
  shrink; the two SF mirrors reuse that workspace and add no second allocation.
- [ ] Confirm all configured FULL captures succeed, including sizes 1..80 (the
  current image reports six FULL sizes, largest 40). Keep async scheduling on.
- [ ] Run `decode_ss` c1/c4 prose and code. Compare XQA with R106 NVFP4 FA2
  baselines: prose `109.9 / 447.8`, code `142.8 / 565.4`. Also retain the same
  engine's FP8 references: prose `131.7 / 578.6`, code `195.8 / 675.4`.
- [ ] Restart with `VLLM_SM12X_NVFP4_XQA=0`; assert
  `decode_backend=flashinfer-native` and repeat the identical decode matrix.
  This is the decisive XQA-vs-FA2 A/B.
- [ ] Run the fidelity ruler against the validated FP8 reference in both XQA
  and FA2 modes. This is mandatory because XQA is a new numerical read path.
  R106's accepted band was top1 `0.8895`, delta-NLL `2.25%`; require parity
  within the established noise rather than boot-only success.
- [ ] Run cold and warm 90K needles, plus the existing deeper-context suite.
  A fluent answer is not sufficient; exact retrieval remains the gate.
- [ ] Record qwen3_5_mtp `ns=4` acceptance, draft/accepted lengths, and decode
  throughput. Confirm q_len=1 logs/metrics use XQA while q_len>1 verification
  retains FA2 and FULL graphs.
- [ ] Exercise `0104` with MTP under the captured sizes and compare acceptance
  and output fidelity with `0104` absent. A capture/replay mismatch or private
  dispatcher failure requires reverting `0104`, not changing the NVFP4 writer.
- [ ] Boot the DFlash draft configuration. `0104` intentionally leaves DFlash
  PIECEWISE/signatures unchanged; keep the R106 per-layer NVFP4 skip fallback
  for incompatible draft attention geometry.
- [ ] Run an FP8/auto control on the patched image. Its backend selection,
  outputs, capture behavior, and throughput must match stock v0.28 within run
  noise.

## Risks and fallback verdict

- No GPU is available here, so XQA JIT loading, workspace sufficiency, CUDA
  graph capture/replay, numerical fidelity, and throughput remain runtime
  gates.
- R106's native SM120 NVFP4 route never allocated the global TRTLLM workspace.
  XQA does. Under the requested FULL configuration, v0.28's graph-memory
  profiler should allocate and charge it before sizing the final KV pool; keep
  `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` enabled and verify the reported
  pool. If graph profiling never exercises q_len=1 XQA (for example eager-only
  or PIECEWISE-only), the first live decode can allocate the workspace after
  pool sizing; use the FA2 env fallback or leave explicit GPU-memory headroom
  if that mode is tested.
- The primary copies the complete K and V SF caches once per XQA layer call.
  It avoids allocation and copies only one ninth of packed NVFP4 bytes, but the
  extra memory traffic can still limit decode. Measure c1 and c4 rather than
  assuming XQA's kernel gain dominates it.
- The lower generated SM120 XQA implementation was not supplied. Compacting SF
  addresses the novel and empirically suspect stride; K/V remains zero-copy
  because the shipped FP8 XQA path already proves support for gapped K/V
  tensor views. A full K/V mirror would duplicate and stream the entire layer
  cache and is not the primary. If fidelity falsifies the SF-only mapping, the
  complete, supported fallback is `VLLM_SM12X_NVFP4_XQA=0` (R106 FA2); do not
  alter the frozen linear writer.
- XQA does not return LSE, so DCP is excluded by the route. This package targets
  the stated single RTX 5090.
- `0104` uses the dispatcher's private padded-descriptor helper and its
  `has_lora=False` capture assumption, matching the old PR. Validate capture
  sizes and MTP acceptance on this exact v0.28 image; treat LoRA and DP as
  separate, unvalidated combinations.
- Literal env value `0` disables XQA; unset or any other value enables it when
  geometry and platform gates pass.

There are no code TODOs or unresolved source-layout choices in the artifacts.
The remaining items above are explicitly GPU-dependent validation gates.
