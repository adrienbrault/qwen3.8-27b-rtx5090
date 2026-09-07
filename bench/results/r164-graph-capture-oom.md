# R164, the graph-capture OOM on the nvfp4 candidate: five pooled FlashInfer wrappers per captured shape; patch 0131 halves graph memory, the pool sizing is the rest (2026-09-03, `results/2026-09-03-r164-bugc`, `results/2026-09-03-r164c-ws`, patch 0131)

[← all results](../RESULTS.md)

R163 left the nvfp4 candidate unable to boot at `--max-num-seqs 32` (OOM during CUDA-graph capture) and marginal at 16. A capture ledger (a diagnostic patch that logs free/allocated/reserved memory around every captured descriptor) found the owner: vLLM sizes the KV pool before graph capture and the profile run skips attention on this path, so all graph memory has to fit in the `1 − util` headroom; on the NVFP4 FA2 route every graph-bound target descriptor creates five `BatchPrefillWithPagedKVCacheWrapper` objects (one per attention-group metadata builder), each with an 8 MiB integer workspace, 40.6 MiB per captured shape, plus 8.1 MiB on the drafter. At SEQS 16 the target manager retained 1,950 MiB against 786 MiB on fp8; graph memory 2.04 GiB against 0.80.

Capping the capture list (`--cudagraph-capture-sizes 10 20 40 80`) boots but costs about 25% at 16 streams on both KV dtypes, so it was rejected. Patch [0131](../../patches-v0280/0131-nvfp4-pooled-int-workspace.diff) keeps a 1 MiB integer workspace on the pooled wrappers (`VLLM_SM12X_POOLED_INT_WS_MIB`, 0 restores upstream's 8 MiB); the planner raises if it is ever short.

| cell (candidate shape, util 0.90) | graph memory | pool | outcome |
|---|---|---|---|
| SEQS 16, before 0131 | 2.04 GiB | 984,411 | boots, first 131K prefill OOMs |
| SEQS 16, 0131 | **1.20 GiB** | 984,411 | boots on the second attempt (first died in the pre-warm), then clean: needles 4/4, benchy c1 244.9 ± 27 / c8 640 / c16 698 t/s, decode_ss code c8 1,027 (128/stream) and c16 1,485 (93/stream) aggregate, 0 preemptions |
| SEQS 32, before 0131 | OOM during capture | — | no boot |
| SEQS 32, 0131 | **1.70 GiB** | 983,314 | boots with 67 MiB free; the first requests OOM |

The pool does not move because it is computed before capture; the fix cuts the graph memory but nothing hands the saving back to the pool. The remaining step is the accounting, and upstream has now done it (R165 above), which is the clean path rather than a per-SEQS utilization table in the launcher. Analysis of the ledger was done with codex (gpt-5.6-sol) on source dumps; every launch and measurement ran here.
