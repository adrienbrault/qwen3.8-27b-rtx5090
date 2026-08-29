# R126: DFlash2 speculator CUDA-graph metadata contract

## Result

`0113-dflash-speculator-graphs-v0280.diff` is a 44-line,
package-root-relative patch against the supplied post-0106
`cur3-dflash2_speculator.py`. It changes only the builder-facing configuration
for the exact failing combination:

- DFlash2 requires non-causal attention;
- draft KV cache dtype starts with `nvfp4`;
- the device is SM12x; and
- the inherited builder mode is exactly `CUDAGraphMode.PIECEWISE`.

For that combination, the speculator gives its attention metadata builder a
private shallow copy of `CompilationConfig` with
`cudagraph_mode=FULL_DECODE_ONLY`. The original engine configuration is not
mutated. Every other path returns the inherited configuration unchanged, so
non-DFlash and FP8 execution retain their existing behavior.

Patch SHA-256:

```text
faae127eae6951e2ced137aef5c63d347e2006996943035f5ad5e9f43d260385
```

All patch paths are relative to the installed `vllm/` package directory; there
is deliberately no leading `vllm/` path component.

## Why this stops the replay-time IMA

The failing ownership chain is split across two graph systems:

1. The engine's configuration remains PIECEWISE.
2. The DFlash query manager nevertheless captures and replays its own FULL
   graph in the measured image.
3. `DFlashCudaGraphManager` correctly asks `build_attn_metadata` for capture
   metadata, but `for_cudagraph_capture=True` does not create a FlashInfer graph
   wrapper by itself.
4. The post-0109 FlashInfer builder allocates its fixed non-causal qo/page
   buffers and selects `_noncausal_prefill_wrappers_cudagraph[batch_size]` only
   when its construction-time mode has FULL decode semantics.
5. With the PIECEWISE engine config inherited by the draft builder, that gate
   is false. Capture therefore closes over the mutable singleton non-causal
   wrapper. A later runtime `plan()` can replace its internal launch metadata,
   leaving replay to dereference state outside the captured contract; the
   observed failure then surfaces at `torch.cuda.graph.replay()`.

The new property is consumed when `DraftModelSpeculator.set_attn()` initializes
the draft attention backend. Changing only that builder-facing copy to
`FULL_DECODE_ONLY` makes 0109's graph wrapper pool and fixed buffers exist
before DFlash capture. The engine remains PIECEWISE and DFlash's own capture
policy is unchanged.

No new key is needed for fixed K. The exact-current capture path uses padded
`BatchExecutionDescriptor.num_reqs`; the runtime metadata build uses the same
padded request count, which is the 0109 wrapper-pool key. The descriptor also
freezes and checks the uniform token width. Adding per-call K or a second key
dimension would be the withdrawn 0110 design, not an R126 fix.

## Reference-hunk mapping

There is no honest fixed-K graph-contract hunk to copy verbatim from the three
Sean diffs. The two 0113 hunks are the minimal v0.28 adaptation at the missing
metadata-builder seam:

| 0113 hunk | Reference mapping | Treatment |
|---|---|---|
| imports at file start | The v0.28 DFlash parent already returns a private outer `VllmConfig`; current FlashInfer uses the SM12x platform gate throughout its NVFP4 route. | Add `copy` to detach the still-shared nested `CompilationConfig`, and `current_platform` to keep the change on SM12x. No execution changes occur in this import hunk alone. |
| `DFlash2Speculator.attn_vllm_config` | `0109` is the concrete fixed-buffer/wrapper contract. Sean's speculator diff instead offers `VLLM_DFLASH_FORCE_EAGER` and adaptive-K plumbing; it does not activate 0109's pool for a PIECEWISE draft builder. | Preserve the inherited non-causal attention config, then expose FULL decode semantics only to the affected builder. This is the graph-preserving v0.28 counterpart to avoiding the integrated graph altogether. |

The named reference diffs are otherwise accounted for as follows:

- `sean-dflash_cudagraph.diff` only derives and forwards per-call K from
  `uniform_token_count`. It is omitted; the fixed-K DFlash2 callback would
  reject that new keyword.
- `sean-cudagraph_utils.diff` adds dispatch logging and an environment-bounded
  FULL-graph capture list. Its default does not repair metadata, while a bound
  merely forces eager fallback. It is omitted.
- `sean-dflash_speculator.diff` bundles adaptive K, per-K buffers, cutoffs,
  logging/NVTX, and eager escape hatches. Those are omitted. In particular,
  the published working profile validates forced eager drafting, not SM120
  NVFP4 FULL replay.

Accordingly, there is no hunk in `dflash/cudagraph.py` or generic
`cudagraph_utils.py`: their exact-current padded request-count acquisition,
fresh capture metadata, descriptor keying, and fixed uniform-width checks are
already sufficient.

## Apply order

Keep the promoted v0.28 stack, then apply in this relevant order:

1. `0106-dflash2-selector-sampling-v0280.diff`;
2. `0109-dflash-noncausal-complete-v0280.diff`;
3. `0113-dflash-speculator-graphs-v0280.diff`.

Do **not** apply the withdrawn `0110-dflash-adaptive-v0280.diff`. The unrelated
0111 and 0112 patches may precede 0113; they do not touch this file.

From the installed package root:

```bash
VLLM_PACKAGE_ROOT=/usr/local/lib/python3.12/dist-packages/vllm

patch --batch --forward --dry-run -p1 -d "$VLLM_PACKAGE_ROOT" \
  < /path/to/out-r126/0113-dflash-speculator-graphs-v0280.diff
patch --batch --forward -p1 -d "$VLLM_PACKAGE_ROOT" \
  < /path/to/out-r126/0113-dflash-speculator-graphs-v0280.diff
python3 -m py_compile \
  "$VLLM_PACKAGE_ROOT/v1/worker/gpu/spec_decode/dflash2/speculator.py"
```

## GPU ambiguity and alternatives

The source contract identifies the missing wrapper acquisition, but only the
ns5 SM120 process can prove that the installed FlashInfer wheel captures and
replays this NVFP4 wrapper safely.

- **ALTERNATIVE A — reference-validated escape hatch:** if the first replay
  still faults, port Sean's small `VLLM_DFLASH_FORCE_EAGER` branch and force the
  drafter eager. This confirms/avoids the integrated graph fault, but it is not
  a CUDA-graph port and is intentionally not shipped in 0113.
- **ALTERNATIVE B — mode disambiguation:** stock v0.28 normally disables the
  DFlash graph manager when its resolved config is exactly PIECEWISE. The brief's
  blocking trace proves the deployed image has a separate FULL replay path. If
  boot inspection instead shows the builder config is `FULL_AND_PIECEWISE`
  (FULL at decode) and only the runtime dispatch says PIECEWISE, 0113's gate is
  correctly a no-op because 0109 is already enabled; do not widen it blindly.
  Inspect wrapper identity/fixed buffer addresses at capture and replay.
- If the deployed fork can run a FULL DFlash manager while its builder mode is
  `NONE`, the narrow condition can be changed to `decode_mode() != FULL`.
  That broader form is not shipped because it also makes explicitly eager
  DFlash2 NVFP4 builders allocate graph wrappers.

None of these alternatives justifies adaptive/per-call-K plumbing for the
fixed-K failure.

## Local verification

One verification pass was run against clean package-shaped `base` and `work`
trees populated directly from all three supplied `cur3-*` snapshots:

```bash
patch --batch --forward --dry-run -p1 -d "$BASE" \
  < out-r126/0113-dflash-speculator-graphs-v0280.diff
patch --batch --forward -p1 -d "$WORK" \
  < out-r126/0113-dflash-speculator-graphs-v0280.diff
PYTHONPYCACHEPREFIX="$TMP/pycache" python3 -m py_compile \
  "$WORK/v1/worker/gpu/spec_decode/dflash/cudagraph.py" \
  "$WORK/v1/worker/gpu/cudagraph_utils.py" \
  "$WORK/v1/worker/gpu/spec_decode/dflash2/speculator.py"
```

- package-root dry-run: **passed**, exit status 0;
- materializing apply: **passed**, exit status 0;
- one `py_compile` invocation over the three exact-current files: **passed**,
  exit status 0.

No tests, lint, imports, model execution, serving, benchmarks, or GPU commands
were run.

## Target-GPU checklist

1. **THE test:** boot ns5 on SM120 with the production PIECEWISE engine config,
   DFlash2 non-causal attention, fixed K, and NVFP4 draft KV. Capture and the
   first real DFlash replay must complete without an illegal memory access.
2. Exercise several captured/padded request counts through repeated replay.
   Confirm non-causal NVFP4 remains on FlashInfer FA2 and no eager fallback or
   wrapper re-planning corrupts an already captured graph.
3. Run the long-context needle check (including the established 40K point) and
   compare the answer with the no-spec/current-good baseline.
4. Run decode checks for both prose and code at the normal concurrency points;
   record correctness, acceptance length, and throughput, and watch for NaNs,
   invalid token IDs, or delayed CUDA faults.
5. Smoke FP8 DFlash2 and a non-DFlash speculator. Both should follow their
   pre-0113 configuration path unchanged.
