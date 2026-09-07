# R190, first GPU numbers for the R190 patch set: FlashInfer B12x is the fastest shared-layout NVFP4 GEMM at every decode M, the served FlashInferCutlass is not (2026-09-05 03:51 to 03:57 UTC, `results/2026-09-05-r190-microbench`, `scripts/r190-microbench.sh`)

[← all results](../RESULTS.md)

GEMM census (`patches-v0290/r190/gemm/nvfp4_gemm_census.py`, one RTX 5090, CUDA-graph replay medians, warm weights, per-rank TP2 shapes: gate_up N=17408 K=5120, down N=5120 K=8704). Median µs at M = 10 / 80 / 160 (the decode batch sizes at c1 / c8 / c16 with 9 draft tokens):

| kernel class | down M=10 | down M=80 | down M=160 | gate_up M=10 | gate_up M=80 | gate_up M=160 |
|---|---:|---:|---:|---:|---:|---:|
| FlashInferCutlass (served) | 44.7 | 36.5 | 38.6 | 44.7 | 26.3 | 42.7 |
| FlashInferB12x | 18.1 | 20.1 | 26.3 | 30.4 | 24.2 | 40.6 |
| Cutlass | 30.4 | 30.4 | 32.4 | 52.9 | 26.3 | 42.7 |
| FlashInferCudnn | 30.0 | 24.2 | 32.4 | 46.8 | 26.3 | 42.7 |
| Marlin (own weight layout) | 14.0 | 48.8 | 81.6 | 20.1 | 75.4 | 132.8 |
| Humming (own weight layout) | 14.0 | 38.6 | 67.2 | 20.1 | 65.2 | 122.5 |

B12x wins or ties at every M up to 256 on both shapes. At c1 the two MLP GEMMs take 48.5 µs per layer on B12x against 89.4 µs on the served kernel; over 56 layers that is 2.3 ms of the 17.1 ms step measured in R187 if it carries into the engine. Marlin and Humming are faster still at M up to 16 but lose from M = 40, so a static Marlin allowlist (patch 0139, R188) can only pay at c1. At prefill sizes (M 2048 and 8192) the FlashInferCutlass and Cutlass classes are within 2%. The engine A/B of patch 0140 (per-layer, per-M dispatch among the shared-layout classes, no second weight copy) is `scripts/r190c-dispatch.sh` with `patches-v0290/r190/gemm/dispatch-b12x.json` and the generated 58-rule table.

GDN spec-update microbench (patch 0145, `patches-v0290/r190/gdn/gdn_spec_microbench.py`, N = 1 / 8 / 16 rows, T = 10, 57 configurations, all pass against the baseline kernel and an fp32 reference): the best N = 1 configuration (tiled, bv 8, 1 warp, 3 stages) runs in 9.2 µs against 11.3 µs for the baseline; at N = 8 and N = 16 nothing beats the baseline by more than 0.2%. With 48 calls per step this is about 0.1 ms per step at c1 and nothing at c8, and every non-baseline configuration differs from the baseline by bf16 rounding. Parked.

Two diagnostics did not run: the fused residual+RMSNorm probe (patch 0144) instantiated a vLLM CustomOp outside a config context and failed before the kernel (probe fixed, rerun `scripts/r190b-fusednorm.sh`); the host-sync census (patch 0142) killed the engine on the first decode because its scope lock rejects the async-scheduling overlap that is the served configuration (fix in progress). The first R188 run failed on every Marlin arm because patch 0139 stored a compiled regex in the vLLM environment table and the AOT-compile hash rejects that type; the patch now keeps the raw string (see `patches-v0290/NOTES22.md`).
