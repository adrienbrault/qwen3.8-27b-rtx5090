# R111: DFlash2 selector-walk sampling fidelity

## Result and scope

`0106-dflash2-selector-sampling-v0280.diff` ports only Sean's selector-walk
sampling core and its degenerate-index guard onto the supplied v0.28.0
`DFlash2Speculator`. It preserves v0.28's fixed speculative-step structure,
buffers, draft-logit lifecycle, cache kernel, model call, and graph plumbing.

There is no `gumbel.py` hunk. The supplied v0.28
`v1/worker/gpu/sample/gumbel.py` already defines `tl_rand32` and `tl_rand64`;
it also already contains the `gumbel_noised_argmax` addition shown in
`gumbel-SEAN-hunks.diff`. The rewritten selector imports the two existing RNG
helpers directly, so changing `gumbel.py` would be redundant.

The previous `0101`, `0103`, and `0104` patches do not name either
`v1/worker/gpu/spec_decode/dflash2/speculator.py` or
`v1/worker/gpu/sample/gumbel.py`, so `0106` has no textual overlap with them.
No GPU-ambiguous implementation choice remains in this narrowly specified
port, so no `ALTERNATIVE` hunk is included.

## Per-hunk rationale

| Patch hunk | Sean reference | Change and expected acceptance effect |
|---|---|---|
| Imports at upstream lines 10-11 | `speculator-SEAN.py` lines 10-11 | Import `tldevice` plus the existing `tl_rand32`/`tl_rand64` helpers and stop importing the wrapper that the rewritten walk no longer calls. `tldevice.log1p` preserves Sean's fp32 Gumbel transform near the winning tail. |
| Kernel signature at upstream line 28 | `speculator-SEAN.py` kernel signature | Remove the compile-time `SAMPLE_PROBABILISTIC` switch. Selection is now based on each request's runtime temperature, allowing greedy and sampled requests to share the same launch without globally forcing one behavior. |
| Score load at upstream line 46 | `speculator-SEAN.py` score load | Keep selector logits in fp32 before temperature scaling, exactly as Sean's core does. With `USE_FP64`, the fp64 random term promotes the sampled expression afterward; this retains the reference arithmetic order instead of pre-promoting logits. |
| Selector core and guard at upstream lines 54-64 | `speculator-SEAN.py` lines 53-78 | For `temperature == 0`, explicitly choose the lowest-index maximum. Otherwise derive a per-position seed with `tl.randint`, draw Gumbel noise with the fp32/fp64 helper selected by `USE_FP64`, and reduce `scores / temperature + noise`. This reproduces Sean's proposal walk for non-greedy requests. The final `tl.minimum(index, top_k - 1)` contains the `BLOCK_K` sentinel produced by all-NaN/all-masked reductions, preventing a one-past-row candidate read and garbage draft token. |
| Launch at upstream line 156 | `speculator-SEAN.py` `_sample_path` launch | Stop specializing the kernel on `self.draft_logits is not None`; the kernel reads each mapped request's temperature dynamically. This completes the signature change while leaving v0.28's proposal-logit cache behavior intact. |

The RNG offset remains the candidate ID, matching the supplied Sean source.
The fidelity change is the direct per-request selector branch, reference
arithmetic/reduction, and numerical containment guard; it does not invent a
lane-offset noise scheme absent from the reference.

## Deliberately excluded Sean changes

The patch does not port Sean's dynamic `num_speculative_steps` argument,
per-K buffer dictionaries, forced draft-logit allocation, profile-run bypass,
candidate-ID clamps, selector-token staging, or related call-chain changes.
Those changes are outside this acceptance-focused work package and would
replace v0.28 structure rather than minimally port the sampling core.

## Apply order

Keep the already promoted v0.28 stack and its writer overlay in place, then
apply the Python patches in this order:

1. `0101-sm120-nvfp4kv-fa2-routing-v0280.diff`
2. `0103-sm120-nvfp4-xqa-decode-v0280.diff`
3. `0104-mtp-drafter-full-cudagraph-v0280.diff`
4. `0106-dflash2-selector-sampling-v0280.diff`

From the installed `vllm` package root:

```bash
cd /usr/local/lib/python3.12/dist-packages/vllm
patch --dry-run -p1 < /path/to/0106-dflash2-selector-sampling-v0280.diff
patch -p1 < /path/to/0106-dflash2-selector-sampling-v0280.diff
python3 -m py_compile v1/worker/gpu/spec_decode/dflash2/speculator.py
```

The `0102` linear-V-scale runtime overlay remains mandatory for the SM120
NVFP4 stack but does not patch this Python package path; retain it according to
the earlier package instructions.

## Risks

- The clamp intentionally maps a degenerate selector row to candidate
  `top_k - 1`. This is a safe in-row fallback, but it cannot restore meaningful
  probabilities to an all-NaN/all-masked row and may lower acceptance for that
  token while preventing an invalid read.
- Sean's explicit reduction chooses the lowest lane among exact ties; that can
  differ from the old helper's backend-specific `return_indices` tie result.
- `USE_FP64` now follows Sean's arithmetic order: fp32 score division followed
  by promotion when fp64 noise is added. This is intentional reference
  fidelity, but it is not bit-equivalent to pre-promoting selector scores.
- Candidate IDs still key the random draws, exactly as in Sean's supplied
  implementation. Acceptance improvement therefore depends on the runtime
  branch/reference reduction and the guard, not on changing RNG offsets.
- Triton compilation/runtime behavior and measured acceptance/throughput
  require the target GPU and are not asserted by the permitted local checks.

## Local verification

Only the two checks authorized by the work package were run:

- `patch --dry-run -p1` against a scratch copy rooted like the installed
  package: passed with exit status 0.
- `python3 -m py_compile` on the scratch file after applying `0106`: passed
  with exit status 0.

No tests, benchmarks, GPU code, model code, lint, import execution, or serving
commands were run.

## Target-GPU checklist

- Boot the patched stack and confirm DFlash2 is selected for the DFlash2 draft
  model without Triton compile/runtime errors.
- A/B the old and new speculator at `temperature=0.6` on the same prose corpus;
  record `decode_ss accept_per_draft` by draft position.
- Repeat the same old-versus-new A/B at `temperature=0.6` on the code corpus;
  record `decode_ss accept_per_draft` by draft position.
- Compare decode throughput under the same workload and concurrency settings.
- Run the 40K needle test and compare correctness with the current baseline.
- If acceptance moves, run the tool evaluation twice and compare both runs
  with the current baseline.
