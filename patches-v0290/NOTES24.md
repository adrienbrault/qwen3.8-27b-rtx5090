# 0141 — packed single-token GDN tuning

## Source finding and scope correction

The brief's ten-row packed recurrence is not present in this source. In
`qwen_gdn_linear_attn.py`, `_forward_core` selects
`_forward_core_decode_non_spec` only when `spec_sequence_masks is None`, there
are no prefills, and there are decodes (lines 1312–1324 in the supplied tree).
That function calls the packed kernel at line 1767. Packed input B is the number
of independent single-token requests. There is no sequence-length or acceptance
argument, and no recurrence loop. Giving ten rows the same state index would race.

Speculative execution instead uses `fused_sigmoid_gating_delta_rule_update`
(lines 1538 onward), or alternate fused MTP / replay-SSM paths. The supplied
`gdn_replayssm_spec_decode.py` does not call this packed kernel. Both the original
non-packed `fused_recurrent_gated_delta_rule_fwd_kernel` and the sigmoid-gating
kernel already keep fp32 recurrence state across rows, loading bf16 once and
storing a bf16 checkpoint for EACH row. They do not reload those rounded
checkpoints between recurrence rows. Acceptance/rollback needs those checkpoints.
A single final store or an unrolled ten-row packed variant cannot be implemented
through the named packed API without changing callers and state semantics.
Those variants are therefore not included, per COMMON's source-impossibility rule.

Consequently this patch is a bounded non-spec packed-path tuning experiment.
It does NOT tune the production ns9 recurrence. A missing proof line during an
ns9 run means the path was not selected; do not attribute any speedup to 0141.
The knob is scoped to calls of the packed launcher, not a global dispatch switch.
No callers, speculative state storage, or acceptance logic are changed.

## Changes and grammar

* `vllm/envs.py`: declare/register `VLLM_SM12X_GDN_DECODE_CFG` as optional string.
* New `ops/gdn_decode_config.py::parse_decode_config`: cached, dependency-free
  full-string parser with explicit supported values.
* `fused_recurrent_gated_delta_rule_packed_decode`: when opted in, validate the
  geometry/platform/dtypes, override launch settings, select the kernel and log.
* New `fused_recurrent_gated_delta_rule_packed_decode_split_kernel`: copy of the
  packed kernel's equations with head-fast flattened CTA addressing.
* `_get_packed_decode_launch_config` and the existing packed/non-packed kernels
  are unchanged. The existing `VLLM_SM12X_GDN_PACKED_BV` remains unset/16/32 with
  its original eligibility and caching behavior. With the new knob set, its
  valid value is superseded by the new config; an invalid old value raises.

Exact ordered grammar (no whitespace, duplicate keys, signs or unknown keys):

```
bv=<n>,bk=<n>,warps=<n>,stages=<n>,split=<n>[,variant=tiled|split]
```

`bv`: 8/16/32/64; `bk`: 128 only; `warps`: 1/2/4; `stages`: 1/2/3;
`split`: 1/2/4, with bv/split >= 8. Omitted variant means `tiled` and requires
split=1. Decimal leading zeros are accepted. Empty string is invalid, not unset.
BK smaller than 128 would omit part of the K reduction; split-K is unsupported.

Two variants:

* `tiled`: original kernel, original V-fast grid, configurable BV/warps/stages.
* `split`: split each nominal V tile into `split` independent CTAs, each with
  effective BV=bv/split; flattened grid maps adjacent CTAs to adjacent heads.
  It has no atomics, scratch tensors, duplicated state elements, or K splitting.
  split=1 also permits isolating CTA order from effective tile size.

Only SM12x with H=8/HV=24/K=V=128, bf16 packed QKV/output and contiguous bf16
state is supported by the override; unsupported combinations raise on launcher
use. Existing launcher shape/device checks still run. As with the original API,
positive state indices must be in bounds and unique per request; these GPU
metadata invariants cannot be checked via host reads during graph capture.
Index 0/negative indices still produce zero output and no state write.

Unset takes the original kernel and exact original launch arguments, including
old BV behavior. No new tensors are allocated. Tuning does not change fp32
recurrence expressions, reductions over the full 128 K elements, or bf16 output
and state conversion points. CTA reordering alone should be bitwise identical.
Different BV/warps may change compiler layouts/reduction trees or contraction;
bitwise identity across those settings is NOT certified. There is no intentional
arithmetic reordering. The benchmark prints bitwise equality and numerical errors.

Example opt-in (set before worker startup; restart to change configurations):

```
VLLM_SM12X_GDN_DECODE_CFG=bv=32,bk=128,warps=2,stages=3,split=2,variant=split
```

Proof, logged via `logger.info_once(..., scope="process")` at the opt-in launch
point (once per fixed configuration per process, hence twice for two ranks):

```
GDN packed decode: variant=split bv=16 bk=128 warps=2 stages=3 (HV=24 K=128 V=128)
```

The proof's BV is EFFECTIVE BV. `docker logs <container> 2>&1 | grep 'GDN packed decode:'`.
A launch proof shows selected configuration, not successful Triton compilation or
that this kernel accounts for all the measured GDN category. The standalone sweep
changes configurations intentionally, so it emits more than one proof line.

## Occupancy and traffic

For one request, original BV32 launches 24*ceil(128/32)=96 CTAs: at most 96/170
SMs receive one CTA in a first wave. BV16 gives 192 CTAs (~1.13 waves), BV8 gives
384 (~2.26 waves); BV64 gives 48. Each CTA holds BV*128 fp32 state elements:
4096/2048/1024 at BV32/16/8, before temporary vectors. At one warp, 4096 state
scalars alone average 128 registers/thread, so register pressure and spills
matter. More warps can reduce per-thread pressure but also increase resource use.
Actual occupancy depends on compiler allocation; CTA count is not occupancy.

For 8/16 requests BV32 already gives 768/1536 CTAs. Smaller tiles can lose due to
redundant Q/K/gating loads and additional CTA overhead; the prior forced BV16 c8
result was -6%. Sweep should find whether BV16 or BV8 with 1–2 warps improves c1,
and whether BV32 with 1–2 warps wins at c8/c16. Stages may do little without a loop.

Each single-token request reads+writes 24*128*128*2*2 = 1,572,864 state bytes/layer.
Across 48 layers: 75.50 MB/request (decimal). But 0.68/1.36/3.22 ms from R186 is
a GDN CATEGORY, including convolution and speculative updates. It is not a
measurement of this packed kernel. Ten-row speculative checkpoint traffic is
also not one read+one write: for the recurrent snapshot path it is one load plus
ten stores, ignoring invalid slots. Thus the brief's roofline inference cannot
be used to establish this kernel's bottleneck.

For the actual served ns9 configuration the expected patch-only GDN times remain
about **0.68/1.36/3.22 ms at c1/c8/c16**, with zero justified improvement until
traces show packed dispatch. As a conditional, optimistic research target ONLY
if the affected update accounts for the category, R186's 25–50% reduction would
mean **0.34–0.51 / 0.68–1.02 / 1.61–2.42 ms**. The 50% endpoint is the best-case
hypothesis, not an offline prediction or measured result.

## Operator procedure and falsification

Build from deliver as context:

```
docker build --network=none -f deliver/Dockerfile.gdn-decode-tuning -t vllm-gdn0141 deliver
docker exec <container> python /opt/gdn_decode_microbench.py --device 0
docker exec <container> python /opt/gdn_decode_microbench.py --device 1
```

The benchmark runs baseline plus 14 configs (both variants, effective tiles
8/16/32/64, warps 1/2/4, stages 1/2/3). Each row count 1/8/16/10/80/160 uses
H=8, HV=24, K=V=128, packed width 5120, bf16 state and normalized Q/K. ALL are
independent single-token rows; 10/80/160 are shape stress tests, not ns9 simulations.
Baseline is both knobs unset. States reset from identical seeded data before
each correctness run and timed graph replay. Random positive state indices are
represented by reversed slot order; a separate NULL-slot check verifies output
zeroing and unchanged state. Outputs AND states are compared to baseline and
to a pure-torch fp32 one-step recurrence. Compile/capture/errors abort, not skip.

Timing is the median CUDA-event kernel-only graph replay time, reset copies
outside the timed interval. The reset warms state caches; reported GB/s is
logical state bytes/us, NOT measured DRAM throughput or full-model bandwidth.
Repeat the sweep in reversed execution/device order if the margin is small.
Default absolute error gates are 0.03125 vs baseline and 0.05 vs fp32, applied
separately to outputs and states; nonfinite differences fail. These are coarse
synthetic screening tolerances, NOT bf16 fidelity promotion thresholds. Use
`--baseline-atol 0` for an exact-equality gate. Random inputs do not certify
long-horizon recurrence stability or extreme gate values.

First inspect dispatch on the served launcher. Run the operator's
`evidence/decode_ss.py --conc 1 8 16 --tokens 1024 --runs 3 --kind code`
and repeat for prose against the local server, with both knobs unset vs finalist.
Use R186 A/B/A/B boots, matched prompts/seeds, acceptance, fixed-work timing and
>=5 seconds of steady decode. Collect traces and run `prof_summary.py` and
`prof_decode_split.py`; inspect raw kernel names too. The latter's step denominator
counts `fused_sigmoid_gating_delta_rule_update_kernel`, so it is NOT valid for a
spec-off run selecting the packed kernel. Use independent iteration counts there.

First number: **packed-kernel calls per target decode step**. If zero under ns9,
the claimed ns9 optimization is falsified immediately. If exercised, require
repeatable >=10% reduction in the affected kernel with no c8/c16 step regression;
no improvement beyond paired variability falsifies the tuning hypothesis. Compare
like effective BV between tiled and split to isolate grid-order benefit.
Finalists must pass R186 protocol F: matched-input bf16 decode rulers at 0/30K,
paired tails/NLL/flips, dense and agentic data, and required acceptance/rollback
checks. Numerical failure rejects a faster setting. 100K+ requires a separately
available reference; none was generated here. No automatic promotion.

## Offline verification and provenance

`bash deliver/verify-0141.sh` copies src/vllm to .work/0141/vllm, dry-runs and really
applies at fuzz 0, checks the real exit status, py_compiles all touched Python plus
benchmark/tests, and runs dependency-free tests. Tests cover grammar, CTA coverage,
unchanged original-kernel AST, identical recurrence AST in the split copy,
identical unset launch arguments for old unset/16/32, and mocked failure paths.
No torch/vllm import, GPU execution, Triton compilation, Docker build, bandwidth,
register/spill measurement, model fidelity or serving latency was verified here.

For THIRD_PARTY.md: the split kernel derives directly from the supplied vLLM
0.29.0rc2 patched `third_party/flash_linear_attention/ops/fused_recurrent.py`.
Its retained header attributes flash-linear-attention, Copyright 2023–2025
Songlin Yang and Yu Zhang, MIT original code; vLLM wrapper carries Apache-2.0 SPDX.
Existing launch configuration derives from local patch 0133, associated in the
supplied R186 evidence with vLLM PR #54181. No exact upstream commit is available
in the dump and none is invented. New parser, grid mapping, harness and tests
are original work for this deliverable. No external code/downloads or network used.
