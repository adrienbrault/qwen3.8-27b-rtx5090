# R116: complete non-causal NVFP4 + adaptive DFlash2 draft length

## Result

- `0109-dflash-noncausal-complete-v0280.diff` completes the post-0105
  non-causal FlashInfer CUDA-graph route.
- `0110-dflash-adaptive-v0280.diff` adds runtime DFlash K plumbing and an
  opt-in, per-request trailing-acceptance controller. It is incremental to the
  post-0106 DFlash2 selector source.

No scheduler API or scheduling-policy change is included. The one shared-runner
hunk only slices the already-produced max-width draft buffer to the active K
before handing it to `DraftTokensHandler`.

## A. Init-time illegal-memory-access diagnosis

The source trace points to the non-causal wrapper's CUDA-graph metadata
contract, not to NVFP4 packing:

1. DFlash marks draft attention non-causal and advertises full-graph support.
2. FlashInfer routes every non-causal batch through prefill.
3. Before `0109`, `_get_prefill_wrapper(causal=False)` returned before the
   causal CUDA-graph branch. It therefore supplied a lazily-created wrapper
   without `use_cuda_graph=True` and without fixed-address qo/page metadata.
4. Engine initialization captures that wrapper. Replay/capture can then observe
   metadata addresses outside FlashInfer's CUDA-graph contract, matching an
   init-time illegal memory access.

`0109` fixes the contract atomically: it advertises non-causal
`UNIFORM_BATCH` support (DCP remains excluded), allocates a persistent
non-causal qo-indptr buffer, pools graph wrappers by padded request count, binds
all fixed metadata buffers, and always supplies the request count to wrapper
selection. The pooled NVFP4 wrapper retains `backend="fa2"` from 0105.

The compared post-0105 source already plans SM12x NVFP4 pages as `torch.uint8`,
keeps the output in model dtype, splits packed pages with
`nvfp4_split_data_scale`, and passes separate K/V data and K/V scale-factor
views to `run()`. Those branches are causality-independent, so adding duplicate
plan/forward plumbing would not address the capture-time fault and is
deliberately excluded.

## B. Controller and runtime-K design

The controller lives in `DFlash2Speculator` and is disabled by default. For
each request it stores the last 32 verified speculative rounds as
`(accepted, drafted)`, where:

```text
accepted = max(num_sampled - 1, 0)       # exclude the bonus token
drafted  = accepted + max(num_rejected, 0)
ratio    = sum(accepted) / sum(drafted)
```

Rounds with no drafts are ignored. The first decision occurs after 32 recorded
rounds; later decisions occur no more often than every 16 recorded rounds. A
strict ratio below `0.35` lowers K by one, and a strict ratio above `0.60`
raises K by one. New requests start at the effective maximum.

DFlash executes one dense query width per launch, while requests retain
independent controller states. The launch uses the minimum desired K among the
active requests, ensuring no request is drafted above its own target without
introducing ragged scheduler semantics.

Runtime K is carried through query preparation, attention metadata, graph
dispatch, model forward, DFlash2 local-convolution block size, selector
buffers, candidate-cache strides, and the scheduler-facing draft tensor. The
candidate cache remains one max-K allocation with an explicit row stride; this
clears the correct old candidate set when K changes and avoids stale
draft-logit columns from per-K caches.

### Environment knobs

| Variable | Default | Meaning |
|---|---:|---|
| `VLLM_SM12X_DFLASH_ADAPTIVE` | `0` | Set to `1` to enable adaptation. |
| `VLLM_SM12X_DFLASH_ADAPTIVE_MIN` | `2` | Inclusive minimum K; positive integer. |
| `VLLM_SM12X_DFLASH_ADAPTIVE_MAX` | `8` | Inclusive maximum K, capped by configured K. |

When disabled, bounds are not parsed and the controller returns the configured
fixed K before reading GPU counters. When enabled, reading the sampled/rejected
counters introduces one opt-in GPU-to-CPU synchronization per proposal call.

## Apply order

From the installed `vllm` package root, retain the earlier stack and apply:

1. `0105-dflash-noncausal-nvfp4-fa2-v0280.diff`
2. `0106-dflash2-selector-sampling-v0280.diff`
3. the externally supplied `0107` prerequisite
4. `0109-dflash-noncausal-complete-v0280.diff`
5. `0110-dflash-adaptive-v0280.diff`

No `0107` artifact exists in this workspace, so its textual overlap could not
be inspected locally. In particular, review any 0107 changes to
`forward_context.py`, the V2 `gpu/model_runner.py`, DFlash CUDA-graph dispatch,
or DFlash speculators before promotion. The supplied diffs otherwise use
package-root paths and `patch -p1`.

## GPU-ambiguous alternative

The recommended patch passes K into captured DFlash callbacks and adds an
ignored optional argument to DSpark because both share the graph manager. If
the external 0107 already provides equivalent runtime-K callback plumbing,
omit the two duplicate hunks in `dflash/cudagraph.py` and
`dspark/speculator.py`; do not stack two implementations. If lower-K graph
descriptors are not captured, dispatch may run those shapes eagerly. That is a
performance question for the target GPU, not a reason to pad back to max K,
which would change DFlash2 convolution/selector semantics.

## Risks and limits

- A mixed batch runs at its minimum per-request K, so one low-acceptance
  request can reduce drafting for its batch peers.
- Adaptation adds an intentional synchronization only when enabled; measure
  whether saved draft work outweighs it.
- CUDA-graph availability for every K depends on the final 0107 graph-size
  policy. The callback plumbing is shape-correct, but lower K may be eager.
- The fixed metadata pool is keyed by padded request count and increases the
  number of persistent FlashInfer wrapper objects.
- Source-only checks cannot establish that the target FlashInfer wheel accepts
  every NVFP4 graph shape, nor can they measure acceptance or throughput.

## Local verification

Only the work-package checks were used:

- `patch --batch --forward --dry-run -p1` against package-rooted scratch bases:
  **pending**.
- `python3 -m py_compile` on every touched candidate Python file: **pending**.

No tests, lint, imports, model execution, serving, benchmarks, or GPU commands
were run.

## Target-GPU checklist

- Boot with adaptive mode off and confirm engine initialization plus DFlash2
  CUDA-graph capture completes with NVFP4 draft KV and no illegal memory access.
- Confirm logs/backend inspection show non-causal NVFP4 prefill using `fa2`.
- Exercise batch sizes that cause graph padding and verify pooled wrappers are
  reused without metadata-address failures.
- Compare adaptive-off output/acceptance with the existing fixed-K stack.
- Enable `VLLM_SM12X_DFLASH_ADAPTIVE=1`; observe K remaining within configured
  and environment bounds and changing by at most one per decision.
- Drive sustained ratios below `0.35`, inside `[0.35, 0.60]`, and above `0.60`;
  verify down, hold, and up behavior after the 32/16 cadence.
- Run mixed-acceptance batches and confirm the launch/returned width equals the
  minimum active request target.
- Exercise K transitions in both directions under probabilistic sampling and
  structured output; check draft-logit rejection fidelity and absence of stale
  tail tokens.
- Confirm lower-K graph/eager dispatch is correct for the final 0107 graph-size
  policy, then measure throughput and the enabled-mode synchronization cost.
