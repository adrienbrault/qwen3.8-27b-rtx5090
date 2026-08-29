# R114: selective GDN kernel port for v0.28.0

## Result

`0108-gdn-kernels-v0280.diff` ports the Sean hunks that remain independently
useful on the supplied v0.28 files:

- carry the active speculative query width (`1 + runtime K`) to both v0.28
  causal-convolution call sites while retaining the configured-width recurrent
  state table;
- preallocate the pure-spec token arange and bypass zero-length or redundant
  CUDA-graph metadata copies;
- bounds-mask the accepted-token-derived initial-state lookup in both FLA
  recurrent kernels.

The patch does **not** port Sean's ReplaySSM-spec feature. Its cache flag, state
shape/dtype calculators, extra cache slots, cursor lifecycle, validation, and
`gdn_replayssm_spec_decode` implementation are absent from the supplied v0.28
base. The launch configurations and call-site fragments in the references are
not a usable feature without that stack.

This is therefore a real but narrower improvement than Sean's overlay. Source
inspection does not transfer its measured flat-per-lane scaling claim to this
patch; the target-GPU measurements below decide the speed verdict.

## v0.28-specific adaptation

Sean narrows each state-index row to `spec_query_len`. That is not carried
over. v0.28 adds a fused CUDA MTP route that passes
`state_indices[:num_requests]` directly to the native operator
(`cur-gdn_linear.py:1750-1758,1798-1812,1865-1876`). A column-prefix view of
the configured-width CUDA-graph buffer is non-contiguous when runtime K is
smaller than configured K. The supplied Python sources do not include the
native operator's stride contract, so changing that layout is not a safe
v0.28 improvement.

Instead, the patch preserves the current full-width, boolean-indexed state
table and supplies runtime K only as `max_query_len` to causal convolution.
It updates both the generic call at `cur-gdn_linear.py:1343` and v0.28's newer
fused-MTP call at `:1721`; Sean's older base had only the first.

The arange allocation is also adapted. v0.28 can clamp
`decode_cudagraph_max_bs` to the capture limit
(`cur-gdn_attn.py:118-125`), but Sean uses the arange before the graph-size
guard. Sizing it to `max_num_seqs * (num_spec + 1)` prevents silent truncation
for an eager batch larger than the capture limit.

## Reference-hunk triage

Every reference hunk is accounted for below. Split rows identify mixed hunks
whose genuine part was separated from ReplaySSM scaffolding.

| Reference hunk / change | Verdict | Evidence and treatment |
|---|---|---|
| `sean-gdn_attn.diff @@ -64`: `spec_query_len` | ported | v0.28 still sizes both spec convolution calls from the configured state width. Add the runtime-width metadata field. |
| `sean-gdn_attn.diff @@ -64`: three ReplaySSM cursors | scaffolding | No current cache flag, cursor lifecycle, extra cache slots, or spec ReplaySSM kernel consumes them. |
| `sean-gdn_attn.diff @@ -144`: persistent token arange | ported | Current pure-spec path allocates `torch.arange` every build at `cur-gdn_attn.py:299-303`. Port with eager-safe capacity rather than Sean's capture-clamped capacity. |
| `sean-gdn_attn.diff @@ -165`: ReplaySSM builder fields | scaffolding | Reads missing `cache_config.use_replayssm_spec` and supports only the omitted ReplaySSM-spec path. |
| `sean-gdn_attn.diff @@ -174`: initialize active width | ported | Initializes the metadata fallback to one query token. |
| `sean-gdn_attn.diff @@ -187`: alternate ReplaySSM sequence mask | scaffolding | Depends on the missing cache-spec feature and its different request classification. |
| `sean-gdn_attn.diff @@ -200`: recover active runtime K | ported | The current builder already receives CPU draft counts; `1 + max(active count)` avoids using configured max K for convolution work. |
| `sean-gdn_attn.diff @@ -252`: active-width token-size multiplier | absorbed | In a pure-spec batch, current `min(product, query_start_loc_cpu[-1])` already returns total actual query tokens; both configured and active maxima bound that total. |
| `sean-gdn_attn.diff @@ -252`: arange and empty-buffer reuse | ported | Removes two per-build device allocations without changing token order. |
| `sean-gdn_attn.diff @@ -252`: compact-front/full-row state slice | scaffolding | Sean's old-base layout shortcut can produce a non-contiguous or wider-stride view on v0.28. Retain current boolean indexing and full width. |
| `sean-gdn_attn.diff @@ -286`: mixed-batch active-width state slice | scaffolding | Runtime width is useful to convolution, not as a change to v0.28's recurrent-state row layout. |
| `sean-gdn_attn.diff @@ -412`: ReplaySSM commit/reset cursors | scaffolding | Imports a missing module and relies on omitted cursor/config/state plumbing; it is not independently callable. |
| `sean-gdn_attn.diff @@ -425`: active-width CUDA-graph state view | scaffolding | The column slice is non-contiguous below configured K and is passed directly to v0.28's new fused native MTP route. |
| `sean-gdn_attn.diff @@ -438`: empty/redundant copy bypass | ported | Current code copies both a zero-length non-spec index and the newly reusable arange into graph buffers. Guard/bypass preserves their addresses and values. |
| `sean-gdn_attn.diff @@ -484`: runtime-K logging | scaffolding | Diagnostic-only environment logging is not a kernel or metadata-path improvement. |
| `sean-gdn_attn.diff @@ -506`: pass active width | ported | Makes runtime K available to both current convolution call sites. |
| `sean-gdn_attn.diff @@ -506`: pass ReplaySSM cursors | scaffolding | Dropped with the omitted cursor feature. |
| `sean-gdn_linear.diff @@ -342`: five-cache state shape | scaffolding | Current v0.28 uses the ordinary four-state calculator; the referenced ReplaySSM-spec calculator and cache contract are absent. |
| `sean-gdn_linear.diff @@ -481`: cache-spec attributes | scaffolding | Reads the missing cache flag and feeds only the omitted custom kernel branch. |
| `sean-gdn_linear.diff @@ -1018`: dtype tuple indexing | scaffolding | `_, state_dtype` and `get_state_dtype()[1]` are identical for the current two-dtype tuple; the edit only accommodates Sean's omitted five-dtype tuple. |
| `sean-gdn_linear.diff @@ -1272`: active convolution width | ported | Ported without narrowing the state table, and extended to v0.28's second fused-MTP convolution call. |
| `sean-gdn_linear.diff @@ -1309`: skip Q/K/V rearrangement | scaffolding | Only Sean's missing packed ReplaySSM kernel accepts `mixed_qkv`; the current FLA recurrent path requires rearranged Q/K/V. |
| `sean-gdn_linear.diff @@ -1372`: ReplaySSM-spec dispatch | scaffolding | The referenced kernel module, five-cache layout, cursor tensors, config, and validation are all absent. v0.28 instead has its supported fused MTP path at `cur-gdn_linear.py:1695-1812`. |
| `sean-fused_recurrent.diff @@ -106`: state-index bounds | ported | Current code dereferences `accepted_tokens - 1` without a row bound at `cur-fused_recurrent.py:103-115`. A masked invalid load returns null block 0 to the existing early exit. |
| `sean-fused_gating.diff @@ -106`: state-index bounds | ported | The same unmasked lookup remains at `cur-fused_gating.py:103-115`; matching guards cover both active FLA recurrent variants. |
| `sean-replayssm.diff @@ -52`: GDN launch tables/helpers | scaffolding | `cur-replayssm.py:3-8,55-65` explicitly implements only `mamba2_output_only`; no supplied current GDN caller requests these keys. |
| `sean-replayssm.diff @@ -63`: GDN config dispatch | scaffolding | Dispatching absent kernels would expose dead configuration keys without changing any v0.28 execution path. |

## Apply order

The four target files do not overlap the known `0101` through `0106`
patches. Retain the `0102` writer overlay, apply the known Python patches in
their existing order, then apply `0108`:

1. `0101-sm120-nvfp4kv-fa2-routing-v0280.diff`
2. `0103-sm120-nvfp4-xqa-decode-v0280.diff`
3. `0104-mtp-drafter-full-cudagraph-v0280.diff`
4. `0105-dflash-noncausal-nvfp4-fa2-v0280.diff`
5. `0106-dflash2-selector-sampling-v0280.diff`
6. `0108-gdn-kernels-v0280.diff`

If an external `0107` is part of the image, apply it in numeric order only if
it leaves these four files at the supplied `cur-*` contents; otherwise
rebase `0108` against that image rather than forcing hunks.

From the directory containing the installed `vllm/` package:

```bash
VROOT=/usr/local/lib/python3.12/dist-packages

patch --dry-run -p1 -d "$VROOT" < /path/to/0108-gdn-kernels-v0280.diff
patch -p1 -d "$VROOT" < /path/to/0108-gdn-kernels-v0280.diff

python3 -m py_compile \
  "$VROOT/vllm/v1/attention/backends/gdn_attn.py" \
  "$VROOT/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py" \
  "$VROOT/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py" \
  "$VROOT/vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py"
```

## Risks

- Runtime K now controls causal-convolution launch width in both generic and
  fused-MTP routes. Query boundaries still come from the existing
  `query_start_loc`, and recurrent state rows remain full-width, but this is
  adjacent to GDN state math and must pass the fidelity ruler.
- For a zero or stale out-of-row accepted-token count, the FLA guards now take
  the existing null-block early return instead of reading another request's
  positive state ID. Valid counts execute the same load and math.
- The persistent arange adds
  `4 * max_num_seqs * (num_spec + 1)` bytes per metadata builder. It removes
  per-build allocation/copy work but is not expected to reproduce the much
  larger gains of Sean's omitted ReplaySSM-spec kernel.
- `py_compile` cannot validate Triton lowering, CUDA-graph address behavior,
  native-kernel contracts, numerical fidelity, or throughput.

## Local verification

Only the two checks authorized by R114 were used:

- `patch --dry-run -p1` against clean package-shaped copies of all four
  supplied `cur-*` files: passed.
- `python3 -m py_compile` on all four scratch files after applying the patch:
  passed.

No tests, imports, lint, model execution, GPU execution, or benchmarks were
run.

## Target-GPU checklist

- Boot the daily configuration and confirm the GDN generic/fused-MTP route
  selected by that configuration compiles and serves without CUDA-graph or
  Triton errors.
- Run `decode_ss` at c1, c4, and c8 on both prose and code. Compare under the
  identical harness/settings with the supplied 124.5, 178, and 1221 baselines;
  record per-stream and aggregate throughput rather than transferring Sean's
  flat-scaling claim.
- Run the fidelity ruler as a promotion blocker. The patch changes the
  convolution's active query window and guards recurrent-state selection.
- Run the 90K needle suite, including concurrent requests, to catch
  cross-request state contamination or deep-context drift.
- Record speculative acceptance and accepted-token distributions alongside
  throughput so a speed gain is not accepted if it comes from lower draft
  quality or skipped state work.
