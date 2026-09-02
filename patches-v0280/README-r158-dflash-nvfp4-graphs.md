# R158: DFlash2 on NVFP4 KV — the drafter belongs in CUDA graphs (0116–0119 revival chain + 0129)

## Result

On the sm120 NVFP4-KV route the DFlash2 drafter was running **eagerly** (0116's `get_cudagraph_support`
returned `UNIFORM_SINGLE_TOKEN_DECODE` for non-causal NVFP4, a precaution carried over from the
eager-drafter/XQA-interference note in seanyourhighness's overlay). A torch-profiler A/B at equal
`num_speculative_tokens=7` / `draft_tensor_parallel_size=1` (RedHat TP2, `scripts/r158-profile.sh`) showed
the whole single-stream deficit vs the fp8-XQA route is **GPU idle time**, not kernel time:

| arm | benchy natural T=0.6 c1 / c8 | GPU busy ms/step | wall span ms/step | attention ms/step |
|---|---|---|---|---|
| nvfp4 + FA2 (drafter eager) | 212.4 / 649.6 | 17.60 | 26.17 | fa2-prefill 0.500 |
| fp8 + dedicated XQA (daily route) | 249.1 / 679.7 | 16.79 | 22.59 | xqa 0.665 |

The FA2-over-NVFP4 verify kernels cost *less* per step than fp8 XQA at 2K context; the eager 5-layer
drafter adds ~50 launches/step and 2.8 ms of idle. `0129` (env `VLLM_SM12X_DFLASH_GRAPHS=1`) re-admits
`UNIFORM_BATCH` for non-causal NVFP4 **and** routes it through the 0109 pooled graph-bound wrappers
(`_get_prefill_wrapper`); the first version without the second hunk faulted (illegal memory access at
the first warmup decode step: the graph closed over the mutable singleton wrapper, the R126 mechanism).

With drafter graphs (`scripts/r158b-graphs.sh`, same settings): **c1 270.7 / c8 678.2**, decode_ss code
c8 1,288, needles 6/6 at 9K/20K/131K, pool 1,029,284 tokens @262K. The sharded drafter
(`draft_tensor_parallel_size=2`, the R155 "Bug A" shape that returned needles 0/4) retrieves 4/4 at
9K/20K under graphs at healthy acceptance; the mechanism is not isolated (consistent hypothesis: an eager
sharded drafter with TP collectives outside graphs on this route). What is established: a 576-cell
differential harness (`scripts/nvfp4_fa2_harness.py`: overlay writer → FlashInfer FA2 paged reader vs an
fp32 reference on the dequantized cache; hd {128,256} × H {2,4,8} × page {16,32,64} × gqa {4,6} ×
causal × q_len × batch × scales) is clean, so the previously suspected FA2 HND-NVFP4 reader
specialization is **not** the bug.

## Files

- `0116-dflash-nvfp4-revival.diff`, `0117-dflash-nvfp4-warmup.diff`, `0118b-dflash-eager-escape-rebased.diff`,
  `0119-dflash-nvfp4-fullgraph-width.diff` — the R155 "revival" chain on top of 0101–0113 (FA2 non-causal
  NVFP4 gate reconciliation + VPE tp1 mask, selector-gather clamps + out-of-compile draft embedding,
  drafter `enforce_eager` escape, DFlash uniform verify width). `Dockerfile.v0280-nvfp4kv-revival`.
- `0129-dflash-nvfp4-drafter-graphs.diff` — env-gated drafter FULL graphs. `Dockerfile.v0280-nvfp4kv-revival-graphs`.
- Constraints that still apply on this shape: XQA **off** (`VLLM_SM12X_NVFP4_XQA=0`, `ALLOW_NO_XQA=1`) and
  `--max-num-batched-tokens 8192` (the R155 "Bug B" dodge: a 256 MiB XQA workspace allocation plus DFlash
  graphs at MNBT ≥ 4929 corrupted FA2 replay retrieval at 262K); FlashInfer workspace 512 MiB.
- Promotion battery at the daily contract (ns9, draft_tp=2, tier on): `scripts/r158c-candidate.sh`;
  numbers in `bench/RESULTS.md`.
