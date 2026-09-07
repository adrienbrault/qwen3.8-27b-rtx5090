# DFlash2 over NVFP4 revived: a working engine after the blocking bugs were fixed (2026-08-31, `results/2026-08-31-r155-revival/`, patches 0116–0119)

[← all results](../RESULTS.md)

The route this repo parked on 2026-08-29 (DFlash2 speculative decoding over an NVFP4 KV cache, where non-causal verify reads appeared to cause an illegal memory access) works again after upstream [vllm#53979](https://github.com/vllm-project/vllm/pull/53979) (plus [#53978](https://github.com/vllm-project/vllm/pull/53978) and [#53977](https://github.com/vllm-project/vllm/pull/53977)) and a night of instrumented debugging. The result: two local bugs found and fixed (three counting the drafter-loader `hasattr` trap caught earlier the same evening), upstream's two warmup OOB fixes applied, two remaining correctness bugs isolated with clean discriminators, and a working engine.

The bugs, in order of discovery:
1. **The historic "IMA" never reproduced.** The 0116 reconciliation (upstream's 47-line non-causal FA2 gate adapted to this tree) plus the #53977 and #53978 warmup OOB fixes boot clean. The old crash was most plausibly those warmup OOBs surfacing asynchronously.
2. **Speculator capture livelock.** The full-cudagraph drafter patch captured unconditionally; a drafter-scoped `enforce_eager` gate (0118) escapes it.
3. **The real livelock: a wrapper-width invariant.** The sm12x graph-bound FA2 prefill-wrapper pool sized itself `1+2N` for any `parallel_drafting` method, but DFlash verifies `1+N`. The mismatch silently deselected the graph-stable wrapper, so FULL capture recorded kernels against a mutable singleton whose plan and workspace mutate before replay, and `CUDAGraph.replay` then spins forever at 100% util/120W. Six-line fix (0119), found via a mid-hang host py-spy dump. Upstream's newer config independently codifies `1+N` for dflash.

Measured envelope (all needles clean, FULL graphs, eager drafter): TP=1 @32K reached code c1 217 t/s at acceptance 0.38; TP=2 (`draft_tp=1`) @32K reached c1 ~207 at 0.34, with prefix caching clean. The capacity gain is measured but not yet usable: **pool 1,183,052 tokens at 262K max-len** (+58% over the promoted fp8+DFlash2 daily's 746,849), behind the second of two isolated correctness bugs:
- `draft_tp=2` corrupts reads (needles 0/4; fp8 KV fine, MTP-over-nvfp4-TP2 fine, which points to the sharded drafter's nvfp4 reads);
- raising `max_model_len` 32K→262K corrupts reads even at shallow depths on the otherwise working shape (max-len-dependent geometry in the nvfp4 addressing).

Method notes: persist engine logs on health-timeout, because a teardown destroyed the first hang's evidence; a host `py-spy dump` at hang time beats config bisection (two blind 25-min boots against one dump that named the exact frame); and needle probes remain the only reliable gate, because every corrupt configuration was perfectly fluent.
