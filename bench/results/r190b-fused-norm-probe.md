# R190b, fused all-reduce + residual + RMSNorm probe (patch 0144): bitwise at the c1 decode shape, one-ulp differences at c8/c16, 9 to 19 µs saved per call (2026-09-05 04:20 UTC, `results/2026-09-05-r190-microbench`, `scripts/r190b-fusednorm.sh`)

[← all results](../RESULTS.md)

Two GPUs, `--atol 0`, H = 5120, ascending then descending M sweep; both ranks and both directions gave the same numbers.

| M | norm bit mismatches | residual bit mismatches | unfused µs | fused µs | saving µs |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 16.2 to 17.2 | 8.3 to 8.8 | 7.4 to 9.0 |
| 10 | 0 | 0 | 22.6 to 23.6 | 10.3 | 12.2 to 13.3 |
| 80 | 3 (one bf16 ulp) | 0 | 58.6 to 60.0 | 42.8 | 15.8 to 17.2 |
| 160 | 4 (one bf16 ulp) | 0 | 94.3 to 94.5 | 74.9 to 75.2 | 19.2 to 19.4 |
| 2048 | fallback | fallback | 1114 to 1119 | 1116 to 1117 | 0 |

A decoder layer runs this sequence twice, so about 112 calls per step; the probe's saving would be about 1.4 ms per step at c1 if it carried into the graph-captured engine, which the eager probe cannot show. The engine A/B is `scripts/r190e-fusednorm.sh` (control, fused, fused repeat; decode probes, greedy decode ruler, agentic ruler).
