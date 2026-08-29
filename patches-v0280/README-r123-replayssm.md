# R123: ReplaySSM-spec for vLLM v0.28

## Result

`0111-replayssm-spec-v0280.diff` is a package-root-relative port of the
ReplaySSM speculative GDN stack onto the supplied post-0108 v0.28 sources. It
adds the OFF-default switch, early validation, K-sensitive compilation hashing,
five-state cache planning/binding, block-keyed cursor lifecycle, circular
chunked verify/flush kernels, Qwen3.5 integration, and v0.28 CUDA-graph routing.

Patch SHA-256:

```text
53a00e8413cb5e54af2976209353ef3b08005aac3f70d419894d6c19f1e83e5e
```

All paths are relative to the installed `vllm/` package directory. They
intentionally do **not** have a leading `vllm/` component.

## Apply and enable

Apply after the existing v0.28 patch stack, including `0108` and the supplied
post-0109 state:

```bash
VLLM_PACKAGE_ROOT=/usr/local/lib/python3.12/dist-packages/vllm

patch --dry-run -p1 -d "$VLLM_PACKAGE_ROOT" \
  < /path/to/0111-replayssm-spec-v0280.diff
patch -p1 -d "$VLLM_PACKAGE_ROOT" \
  < /path/to/0111-replayssm-spec-v0280.diff
```

The feature remains OFF when neither switch is supplied. Enable it at process
startup with either:

```bash
vllm serve MODEL \
  --use-replayssm-spec \
  --mamba-cache-mode none \
  --mamba-backend triton \
  --replayssm-buffer-len 16 \
  ...speculative/MTP arguments...
```

or:

```bash
VLLM_USE_REPLAYSSM_SPEC=1 vllm serve MODEL \
  --mamba-cache-mode none \
  --mamba-backend triton \
  --replayssm-buffer-len 16 \
  ...speculative/MTP arguments...
```

`CacheConfig(use_replayssm_spec=True)` is the equivalent programmatic switch.
Set the environment variable before importing vLLM. Leave
`VLLM_GDN_DECODE_KERNEL` unset or set it to `triton`; an explicit `cuda` value
is rejected because the native fused MTP operator has the ordinary multi-page
state contract.

The enabled configuration is deliberately narrow:

- architecture: `Qwen3_5ForConditionalGeneration`;
- speculative decoding enabled with `K > 0`;
- `mamba_cache_mode == "none"`;
- Triton Mamba backend;
- ordinary `use_replayssm` disabled;
- Mamba cache stochastic rounding disabled;
- no KV-transfer/connector instance;
- `replayssm_buffer_len >= 1 + K`.

These constraints are checked in `VllmConfig` before cache alignment and again
at Mamba-spec construction as a defensive layout check.

## State and cursor contract

Let:

- `B` be `replayssm_buffer_len`;
- `K` be the configured maximum number of draft tokens;
- `T = 1 + K` be the maximum verify window;
- `L = B + T` be the logical flush threshold;
- `R = next_pow2(L) = 1 << (B + K).bit_length()` be the physical ring length.

For the default `B=16`, `K=4`, the logical threshold is `21` and the physical
ring is `32` slots. The single Mamba page per request binds five states in this
exact order:

1. speculative-width convolution state, using the configured conv-cache dtype;
2. recurrent checkpoint `[local_value_heads, value_dim, key_dim]`, FP32;
3. `d` replay ring `[local_value_heads, R, value_dim]`, activation dtype with
   BF16 compacted to FP16;
4. `k` replay ring `[local_key_heads, R, key_dim]`, the same compact dtype;
5. `g` replay ring `[local_value_heads, R]`, FP32.

`num_speculative_blocks` is zero on the enabled path: ReplaySSM stores one
checkpoint page plus compact rings rather than `1 + K` full recurrent-state
pages. Qwen3.5's model-level state calculators use the same five-state tuple so
hybrid attention/Mamba page alignment is correct before layer construction;
the layer spec also promotes stale padding to the actual byte count.

The metadata builder owns three device tensors keyed by physical state block:
`write_pos`, `cache_base`, and `is_flush`. Before each verify, it commits the
previous step's accepted prefix. Verify launches replay committed history plus
the current packed window; flush launches first fold committed history into the
FP32 checkpoint, then evaluate the current window. Both specializations launch
with device-side row routing, retaining CUDA-graph-stable pointers.

Cursor reset covers every non-empty prefill row directly from the original
`is_prefilling` mask, including one-token short-prefill batches and
preemption/recompute onto a reused physical block. A first-decode prompt/context
comparison remains as a fallback. Dynamic `K=0` decode rows still use ReplaySSM;
falling back for such a row would mutate the checkpoint behind a live ring.

## v0.28-specific adaptations

- The `0108` runtime `spec_query_len` is preserved as the actual query width;
  ReplaySSM's state table width is independently fixed at one block.
- A dedicated contiguous `[max_graph_batch, 1]` state-index buffer replaces
  Sean's column-prefix view. This avoids handing a strided narrow view to
  v0.28's newer fused native MTP route.
- The active flag disables that native fused route and forces the custom
  packed Triton branch. OFF retains the original fused/generic decisions.
- Mixed batches pass `a_spec`/`b_spec` to the custom kernel rather than Sean's
  full-batch tensors.
- Cursor reset is derived from raw prefill rows, not final
  `split_decodes_and_prefills` counts, so v0.28's one-token short-extend fast
  classification cannot skip ownership reset.
- Qwen3.5 v0.28 has state-calculator overrides on both the causal-language
  base and `Qwen3_5ForConditionalGeneration`; both are gated. The latter is the
  actual model-registry planning surface for the supported architecture.
- Cache hashing preserves the exact pre-R123 digest when OFF. When ON,
  `use_replayssm_spec`, configured `K`, and the dynamic per-batch-size K
  schedule separate incompatible compilation artifacts.
- `get_replayssm_config` in the supplied tree requires `head_k_dim` for GDN
  selection. The imported kernel wrapper now passes it for verify and flush.

## Source provenance

The local reference bundle contains the GDN call sites and launch tables but
not `gdn_replayssm_spec_decode.py`. The new kernel is pinned to:

- source: <https://github.com/Johnny-Liou/ReplaySSM/blob/a84849410ab56cc2b23432969eb2ecfc42a13d9c/vllm/model_executor/layers/fla/ops/gdn_replayssm_spec_decode.py>
- commit: `a84849410ab56cc2b23432969eb2ecfc42a13d9c`
- blob: `3bc3295e0efa0492484023f53414090a2f0f0081`
- license: Apache-2.0

The only source-level kernel adaptation is supplying `head_k_dim` to the two
launch-config lookups; the v0.28 call-site, cache, and cursor adaptations live
in the surrounding patched files.

Three required current files were not included as `cur2-*` snapshots. They
were reconstructed from immutable upstream v0.28.0 blobs after confirming the
known local `0101..0110` artifacts do not target them:

- `config/vllm.py`: blob `0c90ceff03796df7db1976743cb23e9a0f2811db`;
- `engine/arg_utils.py`: blob `44ee837c6fef64b4807a3f9afc636288fd2f8167`;
- `model_executor/models/qwen3_5.py`: blob
  `28c6a1189937d916785b48abdae13bbc1822d0ed`.

Their patch hunks are small and context-bound. The installation dry-run is the
required guard if the image has an unlisted modification to one of these files.

## Reference-hunk mapping

Every named split reference hunk is accounted for below. “Absorbed” means the
behavior is already present in the supplied post-0108 current file; “omitted”
means it is outside the supported cache-mode-none contract and its consequence
is stated.

| Reference hunk | Verdict | v0.28 treatment |
|---|---|---|
| `sean-config_cache @@ -156` | ported + adapted | Adds the OFF-default field; also accepts the startup env and preserves the old OFF hash. EngineArgs CLI plumbing and early validation come from the consolidated overlay. |
| `sean-mamba_utils @@ -127` | ported | Five dtypes: conv, FP32 checkpoint, compact `d`, compact `k`, FP32 `g`. |
| `sean-mamba_utils @@ -267` | ported | Five shapes with physical `R=next_pow2(B+1+K)`. |
| `sean-mamba_abstract @@ -72` | ported + hardened | Uses zero speculative Mamba blocks when ON; keeps an exact original early return OFF and validates/promotes active page bytes. |
| `sean-replayssm @@ -52` | ported | GDN verify/flush/ordinary launch helpers and Blackwell tables. |
| `sean-replayssm @@ -63` | ported + fixed | Dispatches GDN keys; wrapper supplies the required `head_k_dim`. |
| `sean-gdn_attn @@ -64` | ported | Adds the three persistent cursor tensors; `spec_query_len` was already present from `0108`. |
| `sean-gdn_attn @@ -144` | absorbed | Persistent eager-safe token arange is already in the supplied post-0108 file. |
| `sean-gdn_attn @@ -165` | ported + adapted | Adds active builder constants/cursors and a dedicated contiguous width-one graph index buffer. |
| `sean-gdn_attn @@ -174` | absorbed | `spec_query_len=1` initialization already comes from `0108`. |
| `sean-gdn_attn @@ -187` | ported + hardened | ON classifies every real decode row, requires accepted-token metadata only when decode rows exist, and separately captures raw prefill reset targets. |
| `sean-gdn_attn @@ -200` | absorbed | Dynamic runtime-K recovery already comes from `0108`; ON derives width from actual query lengths so K=0 still routes through Replay. |
| `sean-gdn_attn @@ -252` | adapted | Retains v0.28 boolean row filtering/eager-safe arange but narrows active state rows to one physical block. |
| `sean-gdn_attn @@ -286` | adapted | Mixed-batch state rows likewise use width one while token/query packing remains current v0.28 behavior. |
| `sean-gdn_attn @@ -412` | ported + hardened | Lazy device cursors, commit, first-decode reset, plus raw-prefill reset for preemption and one-token short prefills. |
| `sean-gdn_attn @@ -425` | adapted | Uses the dedicated contiguous width-one graph buffer; never exposes a strided prefix to the native fused path. |
| `sean-gdn_attn @@ -438` | absorbed | Empty-copy and persistent-arange bypasses are already in `0108`. |
| `sean-gdn_attn @@ -484` | omitted | Diagnostic `VLLM_DFLASH_LOG_RUNTIME_K` logging is not execution plumbing and would add a hot-path CPU/list conversion when enabled. No functional consequence. |
| `sean-gdn_attn @@ -506` | ported | Attaches runtime width and three cursor pointers to metadata. |
| `sean-gdn_linear @@ -342` | ported + extended | Adds the five-state shape and dtype override; v0.28 needed dtype routing on the Qwen subclass because the referenced GDN-base current file was not supplied. |
| `sean-gdn_linear @@ -481` | ported | Stores active flag, B, and T; also rejects/guards the incompatible native CUDA MTP route. |
| `sean-gdn_linear @@ -1018` | ported | Indexes recurrent dtype from a five-element tuple. |
| `sean-gdn_linear @@ -1272` | absorbed | Runtime convolution width is already in `0108` and remains distinct from the width-one Replay state row. |
| `sean-gdn_linear @@ -1309` | ported | Skips Q/K/V rearrangement only for the packed custom branch. |
| `sean-gdn_linear @@ -1372` | ported + fixed | Calls the chunked kernel with five caches/cursors and spec-filtered `a_spec`/`b_spec`; disables the newer v0.28 fused MTP bypass. |
| `sean-fused_recurrent @@ -106` | absorbed | Accepted-token/state-index bounds guards are already installed by `0108`; duplicating them would conflict. |
| `sean-fused_gating @@ -106` | absorbed | Same as fused recurrent. |
| `sean-worker_mamba_utils @@ -75` | omitted | Destination/source block bounds guard is general safety work, not Replay cursor/cache plumbing. Cache mode `none` and one Mamba page do not enter state-copy preprocessing. |
| `sean-worker_mamba_utils @@ -104` | omitted | Same: SD-conv copy guard is unreachable under the enforced mode. |
| `sean-worker_mamba_utils @@ -119` | omitted | Same: temporal-copy guard is unreachable under the enforced mode. |
| `sean-kv_cache_interface @@ -231` | omitted | Attention KV-quant-mode string helper is unrelated to GDN Replay pages. No supported-path consequence. |
| `sean-kv_cache_interface @@ -722` | adapted locally | Instead of globally changing every MambaSpec, the enabled branch in `MambaBase` promotes `page_size_padded=max(old,actual)`; Qwen model-level calculators make early hybrid alignment correct. OFF stays exact. |
| `sean-single_type_mgr @@ -1330` | omitted | EAGLE prefix-hit rollback applies to prefix caching; ReplaySSM-spec enforces `mamba_cache_mode=none`. |
| `sean-scheduler @@ -637` | omitted | Structured-output scheduling commentary/behavior is independent of Replay state. |
| `sean-scheduler @@ -651` | omitted | Clearing draft IDs outside the conditional is an ungated scheduler change. |
| `sean-scheduler @@ -1829` | omitted | Supplied v0.28 lacks the referenced `filter_tokens_for_fsm_advance` API; copying it would call a missing method. |
| `sean-scheduler @@ -2159` | omitted | Grammar-aware draft trimming is independent and ungated; structured-output/tool behavior remains a GPU/e2e validation risk. |
| `sean-scheduler @@ -2188` | omitted | Same; raw-draft/statistics semantics are not changed by this cache port. |

Additional supporting hunks recovered from the consolidated Sean overlay are
ported into `config/vllm.py`, `engine/arg_utils.py`, and `qwen3_5.py`: CLI
plumbing, early constraints, Qwen model-level cache planning, and K-sensitive
hashing. The hash port is adapted to run only when ReplaySSM-spec is ON so OFF
retains the pre-R123 compilation digest.

`cur2-flashinfer-post0109.py`, `cur2-worker_mamba_utils.py`,
`cur2-kv_cache_interface.py`, `cur2-single_type_mgr.py`, and
`cur2-scheduler.py` are intentionally not patched.

## Risks and GPU-ambiguous choice

- **Fidelity is the promotion gate.** FP16 replay `d/k` plus repeated history
  reconstruction can diverge from full FP32/BF16 state snapshots. Static
  inspection and Python compilation say nothing about numerical equivalence.
- On SM120, the existing `_is_blackwell()` recognizes only capability family
  100. For the target `K=4` (`T=5`), the lookup therefore uses the conservative
  generic tuple; `T=5` is also not a keyed entry in Sean's even-length
  Blackwell table. Correctness does not depend on the tuned tuple, but the
  measured speedup may not transfer.
- No speculative SM120 launch-table alternative is baked into the patch.
  Guessing that SM100 tuples are optimal on SM120 would make the one artifact
  less safe. The existing `override_replayssm_config` context is the explicit
  alternative for target-GPU sweeps; promote a measured tuple in a later patch.
- The externally sourced Triton kernel has not been lowered or run here.
  Register pressure, shared-memory use, CUDA-graph capture, and compiler
  compatibility remain target-GPU checks.
- Physical cursor ownership is block-keyed. The port resets all prefill and
  recompute rows and excludes graph padding, but preemption/concurrency tests
  remain mandatory.
- The five-state page is larger than one ordinary checkpoint page even though
  it replaces `1+K` full snapshots. Re-profile KV capacity and concurrency.
- The scheduler structured-output hunks are deliberately absent. Grammar/tool
  traffic must pass the tool evaluation twice before promotion.
- The three immutable-upstream baselines named above were absent from the
  supplied snapshots. Any private image edit to them should cause dry-run
  rejection and requires a rebase, not `patch -f`.

## Local verification

Status: pending the single authorized final pass.

The pass will contain only:

1. `patch --dry-run -p1` against clean package-shaped scratch copies;
2. application to a second scratch tree solely to materialize patched files;
3. `python3 -m py_compile` on every touched Python file.

No imports, tests, lint, model execution, Triton lowering, GPU execution, or
benchmarking are authorized or claimed.

## Target-GPU checklist

1. Boot once OFF and once ON with the daily Qwen3.5/MTP configuration. Confirm
   active logs select the Triton GDN path and no page-layout, Triton, or graph
   capture error occurs.
2. Run the **mandatory fidelity ruler** first: OFF versus ON under identical
   seeds/prompts/settings, including near-tie greedy cases, mixed batch sizes,
   dynamic K including K=0, flush boundaries, preemption/recompute, and graph
   padding. Do not promote on unexplained token/logit drift.
3. Run decode c1, c4, and c8 for both prose and code. Compare under the same
   harness with `124.5`, `178`, and `1221`; record per-stream latency,
   aggregate tokens/s, and accepted tokens/s rather than transferring Sean's
   flat-per-lane claim.
4. Run the 90K needle suite, including concurrent requests and enough decode
   steps to cross several flushes.
5. Record MTP acceptance length/rate distributions and dynamic-K selections.
   Reject a throughput gain caused by lower acceptance or skipped work.
6. Run the tool/structured-output evaluation twice, including grammar-constrained
   drafts, because the independent Sean scheduler hunks were not portable.
7. Re-profile KV capacity/max concurrency and compare OFF/ON memory use.
8. If fidelity passes but performance does not, sweep verify/flush launch
   tuples through `override_replayssm_config`; keep SM120 tuning separate from
   this correctness port.
