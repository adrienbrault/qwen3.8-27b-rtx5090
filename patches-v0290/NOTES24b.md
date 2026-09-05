# 0145 — ns9 GDN spec update launch tuning

This follows R186 §3 proposal 4 on the kernel actually selected by ns9. 0141 and
its deliverables are unchanged. Apply 0145 directly to the supplied pcieipc tree;
it does not depend on 0141 or 0139. No caller changes, extra kernels per layer,
scratch tensors, or new state rounding are introduced.

## Functions and variants

* `vllm/envs.py`: optional-string declaration and registration of
  `VLLM_SM12X_GDN_SPEC_CFG`.
* New `ops/gdn_spec_config.py::parse_spec_config`: cached strict parser.
* `fused_sigmoid_gating_delta_rule_update`: opt-in validation and launch override
  ONLY when `num_accepted_tokens` is present. Non-spec portions of mixed steps
  retain their original launch. The string is parsed on either kind of call,
  so malformed strings raise even on a non-spec call.
* New `fused_sigmoid_gating_delta_rule_update_split_kernel`: copy of the original
  decorated kernel with only its program-ID mapping changed. The original
  `fused_sigmoid_gating_delta_rule_update_kernel` is unchanged, including its
  `do_not_specialize`, heuristics, dynamic row loop and all arithmetic.

Two variants are provided:

1. `tiled`: existing kernel/grid, smaller effective BV and configurable warps and
   stages. Priorities: BV16/W2, BV8/W1, and BV32/W2 as a c16 counterpoint.
2. `split`: effective BV=nominal BV/split, flattened head-fast grid. Adjacent CTAs
   visit heads, then V tiles, then requests. Compare to `tiled` at the SAME
   effective BV/warps/stages to isolate order. The original kernel already splits
   V; this variant adds grid ordering, not a new mathematical decomposition.

Checkpoint stores are already contiguous over K, with full masks at K=128 and
V divisible by the chosen BV. `tl.store` does not wait for a host acknowledgement.
There is no evidenced serialization fix from simply dropping masks. Double
buffering a 128×BV fp32 tile increases register pressure and does not remove a
single required rollback write, so it was not implemented. Q/K L2 norms depend
on each row; hoisting all rows requires precomputation in a caller or extra
live vectors. The caller is unchanged. Stages 1/2/3 are compiler launch tuning;
this patch does NOT implement or claim explicit t+1 load prefetch/software
pipelining. That needs GPU compiler/register evidence before adding complexity.

## Knob and eligibility

Ordered grammar copied from 0141, with the new knob name:

```
bv=<n>,bk=<n>,warps=<n>,stages=<n>,split=<n>[,variant=tiled|split]
```

BV 8/16/32/64; BK **128 only**; warps 1/2/4; stages 1/2/3; split 1/2/4;
effective BV>=8. Omitted variant means tiled; tiled requires split=1.
Whitespace, extra keys, duplicate keys, signs, unknown variants and empty strings
raise ValueError. Leading decimal zeroes are allowed. Never split K.

Supported spec calls require SM12x, B=1/H=8/HV=24/K=V=128, bf16 Q/K/V,
contiguous bf16 state [slots,24,128,128], inplace checkpoints, Q/K L2 norm,
non-KDA, varlen offsets, two-dimensional indices with 10 columns and unit token
stride, and enough metadata rows/counts for N requests. Total rows must be
positive and <=10N. Non-spec mixed calls are explicitly outside the override.
GPU metadata is not copied to CPU: per-request lengths 0..10, monotonic offsets,
valid slot bounds, disjoint positive slots across requests and accepted counts
1..10 are caller invariants. The source's NULL/invalid-count guards remain.
Actual row counts can be shorter/zero under padding; T=10 in the proof identifies
the ns9 capacity, not a host inspection of every GPU offset. Other shapes/platforms
on a spec call raise; no silent fallback. Alternate replay-SSM/MTP dispatch that
never calls this wrapper cannot be forced or validated here: require the proof.

Set before worker startup; restart workers to change it. Example:

```
VLLM_SM12X_GDN_SPEC_CFG=bv=32,bk=128,warps=2,stages=3,split=2,variant=split
```

Proof at the launch point, `logger.info_once(..., scope="process")`:

```
GDN spec update: variant=split bv=16 bk=128 warps=2 stages=3 (HV=24 K=128 V=128 T=10)
```

Grep `docker logs <container> 2>&1 | grep 'GDN spec update:'`. BV is effective.
One line per fixed config per process, twice with TP2. The standalone sweep
intentionally changes configs and emits multiple lines. The line proves launch
selection, not successful compilation. Unset uses the same original kernel,
grid, arguments, tensors and numerics. No enabling ENV is set by the Dockerfile.

## Numerical contract

The copied recurrence body is AST-identical. Acceptance chooses checkpoint
`accepted-1` before the loop; every row writes its own bf16 checkpoint; fp32
register state, rather than that rounded checkpoint, carries into the next row.
Sigmoid/softplus, normalization, scale, full K reductions, expression order,
output conversion and checkpoint conversion points are unchanged.

CTA reordering with the SAME BV/warps/stages is bitwise-identical by construction
at the algorithm level: each CTA owns disjoint state columns and full K, with
identical local computation. This still requires compiler/runtime confirmation.
V splitting is mathematically exact and does not intrinsically reorder K sums.
Changing BV or warp count can change Triton's layout/reduction tree or fusion;
therefore no baseline-bitwise guarantee is made for those settings. Changing
stages is likewise checked rather than assumed. No explicit K reduction-order
change appears in the source. Finalists must pass the operator's bf16 decode
ruler, regardless of synthetic allclose results.

## CTA/resource math and conditional payoff

| Effective BV | State fp32 scalars/CTA | CTAs c1/c8/c16 | CTAs/170 SMs c1 | State registers/thread W1/W2/W4 (ideal average) |
| --- | --- | --- | --- | --- |
| 8 | 1024 | 384 / 3072 / 6144 | 2.26 | 32 / 16 / 8 |
| 16 | 2048 | 192 / 1536 / 3072 | 1.13 | 64 / 32 / 16 |
| 32 | 4096 | 96 / 768 / 1536 | 0.56 | 128 / 64 / 32 |
| 64 | 8192 | 48 / 384 / 768 | 0.28 | 256 / 128 / 64 |

The split variant has the same counts at equal effective BV. These are grid
coverage and state-only averages, NOT achieved occupancy or total registers.
BV16/W2 and BV8/W1 preserve the baseline BV32/W4 state scalars/thread (~32).
Smaller BV duplicates Q/K/gate work and may hurt c16, which already has many CTAs.
Per request/layer the state traffic is (1 load + 10 stores)*24*128*128*2 =
8,650,752 logical bytes. V splitting does not reduce that traffic.

An optimistic research target is 10–25% less time in this update, not in the
entire GDN category or target step. Using the supplied profile per-step columns:

| Concurrency | Update baseline ms/step | Conditional tuned ms/step | Saving ms/step |
| --- | --- | --- | --- |
| c1 | 0.42–0.44 | 0.32–0.40 | 0.04–0.11 |
| c8 | 0.96–1.01 | 0.72–0.91 | 0.10–0.25 |
| c16 | 2.27–2.39 | 1.70–2.15 | 0.23–0.60 |

Best-case endpoints assume 25% improvement; they are hypotheses, not measured
predictions. No c16 improvement may be available due to checkpoint bandwidth.
The c16 chunk kernel's ~0.58 ms/step is unaffected. Note a normalization mismatch
in the supplied evidence: 48 calls × 10.0–10.4 µs is 0.480–0.499 ms, not
0.42–0.44; c16 gives 2.573–2.702 ms, not 2.27–2.39. c8 22.8–23.8 µs gives
1.094–1.142 ms. Do not mix the summary's per-step denominator with 48-call math.
For measured per-call savings use **48*saved_us/1000** ms per full target step,
and independently count actual steps in traces.

## Build, benchmark, and decision

```
docker build --network=none -f deliver/Dockerfile.gdn-spec-tuning -t vllm-gdn0145 deliver
docker exec <container> python /opt/gdn_spec_microbench.py --device 0
docker exec <container> python /opt/gdn_spec_microbench.py --device 0 --reverse
```

Run on a free GPU, repeat on device 1 independently. Benchmark uses exact per-rank
N=1/8/16, T=10 shapes, random accepted counts 1..10 and ten permuted disjoint slots
per request. Baseline is the new knob unset (BV32/W4/S3), plus 18 candidates.
First call compiles; failures abort, never skip. It checks outputs and ALL state
slots against baseline, every checkpoint AND final slot against an independent
pure-torch fp32 recurrence, and an untouched sentinel. Graph outputs/states must
be bitwise equal to eager for original and changed acceptance counts, testing
that replay reads GPU metadata. All timing starts from the same state seed;
reset copies are outside event intervals. Eager events include Python dispatch
gaps, while graph events time single-kernel replay. State resets warm caches;
this is not full-model DRAM pressure. Reported speedups use graph medians.

Defaults gate max absolute error separately at 0.00390625 for output/state vs
baseline and vs fp32 (fp32 is not prematurely rounded before comparison).
Nonfinite errors fail. `--baseline-atol 0` requests strict identity. Defaults
are synthetic screening tolerances, not a long-context fidelity policy. The
script exits nonzero on any numeric failure; printed candidate timing winners
are invalid if that happens. Repeat order-balanced runs to resolve timer noise.

First number: **c1 graph-replay µs/call relative to baseline**. If no single
config is >=10% faster (time <=0.90×baseline) at c1 with no c16 loss, this idea
is dead. Harness also requires no c8 loss when printing timing finalists.
Require repeatability; don't promote an isolated sub-microsecond fluctuation.

Then use the existing `decode_ss.py --conc 1 8 16 --tokens 1024 --runs 3 --kind code`
and prose counterpart, matched A/B/A/B boots, same launch settings, prompts,
seeds, contexts and acceptance accounting. Collect traces and run
`prof_summary.py` and `prof_decode_split.py`. For split, update analysis-side
kernel matching to count both `fused_sigmoid_gating_delta_rule_update_kernel`
and `fused_sigmoid_gating_delta_rule_update_split_kernel`, or use independent
step counters (the existing split probe keys the original name). Expect 48
calls per full target decode step and no change in checkpoint/kernel count.
Measure fixed-work target step time alongside throughput and acceptance.
Finalists must pass R186 protocol F bf16 decode rulers at 0/30K, NLL/tails/flips,
code/prose/agentic prompts, and rollback/acceptance checks before promotion.

## Offline verification and provenance

`bash deliver/verify-0145.sh` copies the reference into `.work/0145/vllm`, dry-runs
and applies at fuzz zero, checks the actual apply exit code, py_compiles all
touched Python plus harness/tests, and executes dependency-free tests. Tests
cover grammar rejection, complete nonoverlapping grid mapping, unchanged
original kernel AST, identical copied body/decorators, unset spec/non-spec
launch arguments, override launch settings, and unsupported combinations.
No torch/vLLM imports occur during verification.

Not verified: Triton compilation, register allocation/spills, actual GPU
bitwise identity, launch/capture performance, Docker build, served metadata
padding, model fidelity, throughput or long-context behavior. The operator must
verify these on the serving box. No network or GPU was used for this delivery.

THIRD_PARTY.md provenance: the split kernel is derived from the supplied patched
vLLM 0.29.0rc2 `third_party/flash_linear_attention/ops/fused_sigmoid_gating.py`,
including local accepted-index guards. Its existing header attributes original
flash-linear-attention code, Copyright 2023–2025 Songlin Yang and Yu Zhang, MIT;
vLLM wrapper SPDX is Apache-2.0. No exact upstream commit is in the dump, and none
is claimed. Parser grammar follows the locally delivered 0141. Grid mapping,
validation, harness and tests are original work for this deliverable; no external
code or downloads were used.
