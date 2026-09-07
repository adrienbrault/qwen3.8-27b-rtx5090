# R163, nvfp4 KV candidate vs the fp8 daily, paired: fidelity, pool and c8 gates pass; single-stream jitter and a graph-capture OOM block promotion (2026-09-03, `results/2026-09-03-r163-paired`, `scripts/r163-paired.sh`)

[← all results](../RESULTS.md)

The R158c candidate (RedHatAI NVFP4 weights, **nvfp4 KV**, DFlash2 ns9 TP=2 with drafter FULL graphs: patches 0116–0119 + 0129, `VLLM_SM12X_DFLASH_GRAPHS=1`, XQA decode off, FlashInfer workspace 512 MiB) against the fp8-KV daily shape, same box, same hour, same boot order and probes, both on :8029 at util 0.90, SEQS=8, MNBT 8192, disk tier on (wiped before each arm). Gates are the promotion sheet's.

| gate | fp8 daily shape | nvfp4 candidate | verdict |
|---|---|---|---|
| pool, 3 boots | 629,840 / 627,756 / 627,756 (bimodal, known) | 985,507 ×3 | +57%, no bimodality |
| needles 9K/20K/131K/220K ×2 | 8/8 | 8/8 | pass |
| tier warm-revisit 32K | 7.48 s → 0.45 s | 7.58 s → 0.68 s | pass |
| tool-eval 69×4 | 88.5 ±3.1 | 89.8 ±0.5 | pass |
| llama-benchy c8 (T=0.6, pp2048 tg256) | 636.7 ±13.1 | 648.2 ±3.1 | parity |
| decode_ss code c8 | 1,167.5 (acc 0.288) | 1,153.1 (acc 0.298) | parity |
| decode_ss prose c8 | 955.2 | 869.2 | −9% |
| decode_ss prose c1 @30K ctx | 142.2 [141.6, 142.8] | 142.9 [132.3, 153.4] | parity, jittery |
| llama-benchy c1 | 276.5 ±1.2 | 213.8 ±25.2 | **fail, −23% and jittery** |
| boot at SEQS=32 | pool 601,021; c8/16/32 = 643/640/667 | **OOM twice** | **fail** |
| engine error lines | 0 | 0 | pass |

Extras on the candidate (SEQS=8, needles 131K+220K all hit): ns7 pool 1,030,986, c1 237.8 ±26.8, code c8 **1,262** (+9% over ns9); ns11 pool 946,928, c1 265.7 ±13.0, code c8 1,045. Fewer speculative tokens buys c8 on this KV; the single-stream jitter is present at every ns. A drafter-fp8-under-nvfp4-target cell is not viable: the pool collapses to 290K and the drafter's KV update crashes on the layout mismatch (`ValueError` in `do_kv_cache_update`); mixed KV dtypes are not a supported cell on this overlay.

**The graph-capture OOM ("Bug C").** The candidate's pool does not shrink with `--max-num-seqs` (985,507 / 984,411 / 983,314 at 8/16/32; fp8 drops 627,756 → 601,021) and the engine OOMs at util 0.90: SEQS=16 in the first 131K prefill, SEQS=32 during FULL graph capture with 8.5 MiB free. Source-grounded mechanism: vLLM sizes the KV pool before graph capture and its pre-capture graph estimate is hard-coded to zero, so no graph reserve is subtracted (upstream behaviour, fp8 survives because its graphs, 0.48 GiB at SEQS=8 and 1.11 at 32, fit inside the util headroom); the profile run skips attention on both target and drafter, so the graph-bound non-causal wrapper path is never profiled; the candidate's graphs cost about 2.4x fp8's per captured shape (1.15 GiB at SEQS=8, 2.05 at 16). Both graph managers share the global graph pool, so private pools are ruled out; whether the retained memory is FlashInfer per-wrapper plan storage or capture-time storage is what the next round measures with a per-descriptor allocation ledger (`patches-v0280/0130-bugc-capture-ledger.diff`, env-gated, a no-op by default) and a zero-patch cap on captured shapes (`--cudagraph-capture-sizes 10 20 40 80`, which bounds both managers). Interim workaround: util 0.86 at SEQS=16 (pool ≈ 840K).

Instrument notes: the 262K needle depth is a probe bug (prompt plus answer exceed 262,144 → HTTP 400 on both arms; the true edge needs ≈258K); decode_ss finds no steady-state window at c32 with 512-token generations, use llama-benchy peaks there; the fp8 shape's c1 was 276.5 today against 318.8 on 09-02 with clocks and the memory OC intact, so a jitter-isolation ladder (tier off, eager drafter, CPU governor performance, fp8 control) runs next. Status: fidelity and capacity gates pass; promotion is blocked on the single-stream jitter and on Bug C, both in flight; the candidate's own SWE-Bench Verified run (same 500, util 0.86) is queued to pair per-instance against R160.

## R163c: the single-stream gap is a box-level bimodality, not a candidate defect (`results/2026-09-03-r163c-c1-jitter`, `scripts/r163c-c1-jitter.sh`)

Five fresh boots, llama-benchy c1 ×5 with per-run values kept:

| cell | c1 mean | per-run tg t/s |
|---|---|---|
| nvfp4, tier on (the candidate) | 297.7 ±30.3 | 326.6, 295.9, 240.8, 306.8, 318.5 |
| nvfp4, tier off | 274.0 ±29.7 | 298.8, 241.5, 284.5, 308.8, 236.4 |
| nvfp4, eager drafter | 227.3 ±23.2 | 267.6, 238.0, 215.5, 211.5, 203.8 |
| nvfp4, CPU governor performance | 255.9 ±22.9 | 253.0, 285.7, 244.7, 275.0, 220.9 |
| fp8 daily shape, CPU governor performance | 289.0 ±21.9 | 270.0, 314.9, 281.1, 315.1, 263.9 |

Every cell, fp8 included, alternates within one boot between a high mode (nvfp4 300–327, fp8 315) and a low mode (nvfp4 236–245, fp8 264–281); the paired battery had sampled three low-mode runs on each side. The tier, the drafter graphs (eager is simply slower, as R158 found) and the CPU governor do not remove it. At peak the candidate is +4% over fp8 (326.6 vs 315.1), consistent with R158/R158c. The load average sat at 12–15 during every single-stream cell (the engine's busy-polling threads on a 16-thread CPU); whether the low mode is a GPU clock/power state on one of the two cards or content-driven acceptance variance at T=0.6 is being correlated with a per-second GPU clock/power sampler. The single-stream gate is therefore parity at peak, and the jitter is tracked as a daily-shape issue in its own right. The remaining promotion blocker is the graph-capture OOM.
