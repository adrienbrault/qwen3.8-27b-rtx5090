# R124: gated XQA speculative verification on SM120 NVFP4

## Result

The supplied FlashInfer 0.6.16.post3 contract supports uniform, linear
multi-token verification with NVFP4 KV on SM120. The result is therefore the
gated patch `0112-xqa-verify-v0280.diff`, not an unsupportable verdict.

The patch adds one opt-in switch:

```text
VLLM_SM12X_XQA_VERIFY=1
```

The default is off. Only the exact value `1` enables the new route, and the
value is cached while metadata builders are initialized. Restart the engine
between A/B arms.

With the existing `VLLM_SM12X_NVFP4_XQA` switch left enabled, routing is:

| Batch | `XQA_VERIFY` unset/`0` | `XQA_VERIFY=1` |
|---|---|---|
| SM120 NVFP4, `q_len=1` | Existing 0103 XQA route | Same existing XQA route |
| SM120 NVFP4, positive uniform `q_len=1+ns` | Existing FA2 prefill/verify route | XQA speculative route |
| SM120 NVFP4, ragged or zero-padded multi-token Q | Existing FA2 route | FA2 fallback |
| Other cache/device families | Existing route | Existing route |

`VLLM_SM12X_NVFP4_XQA=0` remains authoritative: it disables the underlying
NVFP4 XQA decode route, so setting the new verify switch cannot bypass the FA2
fallback. DCP is likewise still excluded by the 0103 XQA gate.

## Contract evidence

### Uniform query shape

`r107-src/fi_xqa.py:188-213` exposes `q_seq_len` and an optional
`q_cu_seq_lens`, but no uniform-Q indptr. Its documented speculative shape is
`[batch, beam=1, q_seq_len, num_q_heads, head_dim]`
(`fi_xqa.py:218-221`), and `q_seq_len > 1` selects speculative mode
(`:282-285`). The cumulative offsets at `:293-300` are specifically the
ragged-Q contract; they are neither needed nor passed for this uniform route.

The public decode adapter accepts vLLM's flat query
`[batch*q_len, heads, dim]` (`r107-src/fi_decode.py:3549-3551,3597-3600`). At
`:3685-3691` it views that storage as `[batch,q_len,heads,dim]` and inserts the
beam dimension, producing the exact XQA shape. It forwards `q_seq_len` at
`:3721`.

The patch treats adjacent differences in vLLM's CPU `qo_indptr` only as proof
of the fixed width. A batch is admitted only when all routed request lengths
are the same positive `Q > 1` and the total row count is exactly `batch*Q`.
The existing `split_decodes_and_prefills(..., require_uniform=True)` remains in
force because NVFP4 is deliberately still excluded from `use_dedicated_xqa`.
The extra exact check rejects the split helper's permitted zero-length CUDA
graph padding; such a batch cannot be represented by XQA's uniform API and is
returned to the existing all-prefill FA2 route.

There is one shipped-adapter shape wrinkle. For caller-provided output,
`fi_decode.py:3695-3698` only inserts the beam dimension; it does not first
reshape a flat `[batch*Q,H,D]` output as it does the query. XQA documents that
output matches query (`fi_xqa.py:245-247`). The patch therefore passes a
zero-copy `[batch,Q,H,D]` output view for `Q > 1`; FlashInfer's unsqueeze then
produces `[batch,1,Q,H,D]`. The underlying vLLM output storage is unchanged.

### Draft mask

For `Q > 1`, XQA accepts a bit-packed `uint16` mask with shape
`[batch,Q,2*ceil(Q/32)]` (`fi_xqa.py:286-292`) and rejects a missing mask
(`:416-418`). The current backend already implements this contract:

- `_make_xqa_draft_block_mask` builds each linear causal row with
  `kv_position <= query_position` and packs it into two 16-bit words per
  32 draft positions (`cur2-flashinfer-post0109.py:123-146`).
- `_get_decode_mask` expands and caches that request mask at a stable address
  for the batch (`:1188-1208`). For the target `Q=1+4=5`, the resulting shape
  is `[batch,5,2]`.

The JIT turns on `SPEC_DEC=1` for `Q > 1` and explicitly selects non-tree,
linear draft indexing (`r107-src/fi_jit_xqa.py:106-119`). This supports the
target causal MTP chain; it is not evidence for a tree-draft mask.

### NVFP4 and scale factors

XQA documents packed `uint8` K/V with last dimension `head_dim/2` and separate
K/V scale tensors with last dimension `head_dim/16`
(`fi_xqa.py:222-237`). It explicitly limits that combination to compute
capability major 12 and requires both scale tensors (`:394-400`). Module
selection includes both KV dtype and `q_seq_len` without a mutual exclusion
(`:404-414`), and the launch forwards K, V, both scale tensors, `q_seq_len`,
and the mask together (`:430-455`).

The JIT evidence agrees: `uint8` selects NVFP4 cache enum 3
(`fi_jit_xqa.py:73-80`), the speculative flags are added independently at
`:106-124`, and SM12 is an allowed compilation target at `:127-130`. Its URI
is Q-specific (`:151-165`), so `Q=5` creates a speculative specialization but
does not select a different cache representation.

The patch intentionally stays on the generic dispatcher used by 0103. That
path validates the E4M3 K/V scale tuple (`fi_decode.py:3231-3257`) and forwards
the scale tuple plus mask to XQA (`:3284-3326`). In the vLLM source it therefore
continues to consume:

- the same zero-copy `nvfp4_kv_data` views;
- the same graph-stable compact scale mirrors produced by
  `_prepare_nvfp4_xqa_sf`; and
- the same `get_xqa_bmm1_scale`/`bmm2_scale` values as 0103 single-token decode.

The lower-level XQA docstring calls scale bytes `uint8`, while the public
dispatcher requires `torch.float8_e4m3fn`. This is an existing representation
wording mismatch: both are one-byte containers, and 0103 already exercises
the dispatcher's E4M3 form. No scale view, copy, or writer layout changes here.

## Per-hunk rationale

| Base location | Change | Reason |
|---|---|---|
| `:120` | Add cached, exact-value `VLLM_SM12X_XQA_VERIFY` helper, default off. | Makes the A/B one environment variable and keeps opt-in state static for graph capture. |
| `:893` | Derive `use_xqa_nvfp4_verify` from the existing 0103 NVFP4 XQA flag. | Inherits family-120, NVFP4, head-size, XQA availability, no-DCP, and old env gates. |
| `:931` | Advertise spec-as-decode only when the new gate is on. | Raises the reorder threshold to configured `1+ns`; off retains threshold 1 and FA2 verify. |
| `:1469` | Prove positive uniform Q and exact `B*Q`; send zero/ragged attempts back to all-prefill. | Uniform XQA has no per-request offsets. This prevents padded rows from being reinterpreted as requests. |
| `:1841` | Store Q and the cached causal draft mask in decode metadata. | Supplies XQA's required speculative arguments without enabling the ragged dedicated branch. |
| `:2701` | Replace the blanket multi-token rejection with metadata/shape assertions and a zero-copy 4-D output view. | Enforces the proven contract and repairs the adapter's caller-output shape before its beam unsqueeze. |
| `:2744` | Forward the metadata mask through the existing generic XQA call. | Reuses 0103's NVFP4 data, compact SF, workspace, scaling, and output route. |

No FA2 prefill/verify call, wrapper plan, KV writer, packed cache view, compact
SF preparation, q_len=1 call, FP8 route, or non-SM120 route is modified.

## Apply order

This diff is against the exact supplied
`r123-src/cur2-flashinfer-post0109.py`, representing the installed backend
after 0101 + 0103 + 0105 + 0109. Apply it after that stack from the installed
vLLM package root; its patch path is
`a/v1/attention/backends/flashinfer.py`.

```bash
VLLM_PACKAGE_ROOT=/usr/local/lib/python3.12/dist-packages/vllm

patch --batch --forward --dry-run -p1 -d "$VLLM_PACKAGE_ROOT" \
  < /path/to/out-r124/0112-xqa-verify-v0280.diff
patch --batch --forward -p1 -d "$VLLM_PACKAGE_ROOT" \
  < /path/to/out-r124/0112-xqa-verify-v0280.diff
python3 -m py_compile \
  "$VLLM_PACKAGE_ROOT/v1/attention/backends/flashinfer.py"
```

For the A/B, leave `VLLM_SM12X_NVFP4_XQA` unset/enabled in both arms and
restart between:

```bash
export VLLM_SM12X_XQA_VERIFY=0  # FA2 verification control
export VLLM_SM12X_XQA_VERIFY=1  # XQA verification candidate
```

## Risks

- Source inspection proves API support, not correctness or speed of the
  generated SM120 binary. The Q-specific JIT specialization must build,
  capture, and replay on the target wheel/GPU.
- Only uniform, positive linear-chain Q is enabled. Ragged queries,
  zero-length graph padding, and attempted non-uniform multi-token groups fall
  back to FA2. That fallback is correct but may reduce the performance win at
  sparsely occupied graph sizes.
- The output-view adaptation is source-required and allocation-free, but the
  shipped lower wrapper is not available here for GPU execution. Treat any
  output-shape assertion or incorrect row ordering as a hard failure.
- The existing 0103 workspace/SF scratch capacity and XQA no-LSE restrictions
  still apply. This patch adds no memory allocation in `forward`.
- Sliding-window XQA assumes consecutive linear draft rows; tree drafts with a
  sliding window are explicitly outside the contract (`fi_xqa.py:268-271`).
- The env value is cached. Changing it without restarting does not constitute
  a valid A/B.

## Local verification

Only the requested source checks were performed:

- `patch --batch --forward --dry-run -p1` against a package-rooted scratch
  copy of the exact supplied current backend: passed.
- `python3 -m py_compile` on the patched candidate backend: passed.

No imports, unit tests, lint, GPU/model execution, serving, or benchmarks were
run.

## Required GPU checklist

- Boot once with `VLLM_SM12X_XQA_VERIFY=0` and once with `=1`, restarting each
  time. Keep the existing 0103 XQA switch enabled. Confirm initialization,
  Q-specific JIT build, every configured CUDA-graph capture, and replay all
  complete without fallback/errors.
- Run decode c1 **code** with MTP enabled in both arms. Record throughput,
  accepted tokens per step, accepted/drafted ratio, and any output mismatch.
  Confirm the off arm uses FA2 verify and the on arm launches speculative XQA.
- Repeat the same c1 code prompt with MTP disabled in both arms. Results and
  q_len=1 XQA behavior should be unchanged by the new knob.
- Exercise full and partially occupied graph batches. Positive uniform rows
  should use XQA; zero-padded/non-uniform rows should take the documented FA2
  fallback without assertion or row corruption.
- Run the existing MTP acceptance check and one GSM8K sample. Acceptance must
  remain in the expected band and the GSM8K answer must be correct.
- A full cache-layout fidelity ruler is not required for this patch: both A/B
  arms retain the frozen writer, and the candidate reads the identical
  `nvfp4_kv_data` plus compact K/V scale mirrors through the same XQA-NVFP4
  dispatch used by 0103 q_len=1. The new numerical surface is the speculative
  Q specialization and causal mask, covered by acceptance plus GSM8K. Escalate
  to the full ruler if either sanity check moves.
