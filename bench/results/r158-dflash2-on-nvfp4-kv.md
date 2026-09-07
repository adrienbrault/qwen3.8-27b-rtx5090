# R158, DFlash2 on NVFP4 KV: drafter graphs close the single-stream gap (2026-09-02, `results/2026-09-02-r158-nvfp4-profile`)

[← all results](../RESULTS.md)

**Correction (R183, 2026-09-04):** the NCCL cost reported in this section (16 ms per step, 35% at 8 streams) was measured on llama-benchy prefill chunks, whose all-reduces exceed the custom all-reduce cap; decode all-reduces never go through NCCL. The decode-only profile is in [R183](r183-decode-profile-levers.md).

All arms used RedHatAI NVFP4 weights, TP=2, `num_speculative_tokens=7`, `draft_tensor_parallel_size=1` (identical drafter), MNBT 8192, util 0.90, no tier; llama-benchy natural T=0.6 pp2048 tg256, runs 3. Profile = torch profiler, 60 decode iterations, rank 0 (`scripts/prof_summary.py`, `scripts/prof_cpu.py`).

| arm | c1 | c8 | GPU busy ms/step (c1) | wall ms/step (c1) | attention ms/step | pool @262K |
|---|---|---|---|---|---|---|
| nvfp4 KV, FA2, XQA off, drafter **eager** (0116) | 212.4 ±21.5 | 649.6 | 17.60 | 26.17 | fa2-prefill 0.500 | 1,030,418 |
| fp8 KV, dedicated XQA (the daily route) | 249.1 ±5.0 | 679.7 | 16.79 | 22.59 | xqa 0.665 | 617,079 |
| fp8 KV forced through FA2 (kernel-time control, piecewise) | (188.5) | (664.3) | 20.17 | 30.82 | fa2-prefill 0.278 | 617,079 |
| nvfp4 KV, FA2, XQA off, drafter **graphs** (0129) | **270.7 ±8.5** | **678.2** | — | — | — | 1,029,284 |
| same, `draft_tensor_parallel_size=2` (the R155 Bug-A shape) | **276.9 ±18.0** | — | — | — | — | 1,029,284 |

GEMM (10.2 ms/step) and NCCL (2.1–2.4 ms/step; 16 ms/step = 35 % at c8 on every route) are identical across arms. The nvfp4-vs-fp8 gap at equal drafter was 0.8 ms of GPU time and 2.8 ms of idle per step: the eager drafter issues ~50 extra launches per step. With drafter graphs the nvfp4 shape is +8.7 % faster single-stream than the fp8-XQA route at the same settings, and at parity at c8, with +67 % pool. decode_ss measured code c8 1,287.8 (daily fp8 ns9: 1,212). Needles were 6/6 at 9K/20K/131K (draft_tp=1) and 4/4 at 9K/20K (draft_tp=2); 250K probes returned HTTP 400 (filler over max-len). The 576-cell FA2-NVFP4 differential harness (`scripts/nvfp4_fa2_harness.py`) is clean at 0.32 % mean-rel error (fp4 noise), including the previously suspected hd128/H4/page32 cell. Ledger correction: the R157c claim that ~19 points were "q_len=8 verify on FA2-over-nvfp4" is withdrawn. That kernel is cheaper than fp8 XQA at 2K context.

## R158c: the nvfp4 candidate at the daily contract (ns9, `draft_tensor_parallel_size=2`, drafter graphs, tier on, util 0.90; `scripts/r158c-candidate.sh`)

| gate | nvfp4 candidate | fp8 daily |
|---|---|---|
| pool @262K | **984,959** | 654,491 |
| needles 9K/20K/131K/220K ×2 | **8/8** | 9/9 |
| warm-revisit 32K (disk tier) | 7.58 → 0.66 s | 7.49 → 0.45 s |
| llama-benchy natural T=0.6, c1 / c8 | 302.7 / 655.6 | 318.8 / 656 |
| decode_ss c8 code / prose, c1 @30K | 1,144 / 875 / 146.7 | 1,212 / 925 / 157 |
| tool-eval 69×4 | 89.0 ±1.4 | 90.8 ±0.5 |

The candidate answered correctly to 220K with the sharded drafter under graphs and zero engine errors. The trade is about 5 % decode and a noise-level tool-eval (tool-calling benchmark, tool-eval-bench) delta for +50 % pool. The harness in `--deep` mode (cache extent 1.9/2.4 GiB, referenced pages above the 2^31-byte boundary) is clean in 24/24 cells, so the R155 "Bug B" corruption is not a plain 32-bit offset in the eager FA2 reader. XQA stays off and MNBT stays 8192 on this shape. Not promoted. `scripts/serve-nvfp4-candidate.sh` is the one-command flip.
