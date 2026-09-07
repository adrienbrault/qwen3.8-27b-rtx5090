# R184, an all-reduce microbenchmark at the served decode shapes, NCCL vs vLLM's custom all-reduce vs FlashInfer's pcie_ipc kernel on the two RTX 5090s: pcie_ipc is 36% faster than the served kernel at 10 rows and 24% at 160, the served kernel is slower than NCCL from 80 rows up, and all three sit at the PCIe floor at 80 MB (2026-09-04, `results/2026-09-04-r184-arbench`, `scripts/ar_bench.py`, `scripts/ar-bench.sh`)

[← all results](../RESULTS.md)

**Why.** R183 left one lever for the two-card decode tax: the all-reduce kernel itself. vLLM's custom all-reduce runs every TP=2 all-reduce below its 8 MiB cap, which is every decode step; NCCL runs the prefill chunks above the cap. FlashInfer merged a third option on 2026-08-20, [PR #4393](https://github.com/flashinfer-ai/flashinfer/pull/4393), a `pcie_ipc` all-reduce written for boxes without NVLink and benchmarked there against NCCL. It is in FlashInfer main only, not in any 0.6.x release, and vLLM has no caller for it. R184 measures the three kernels side by side on this box at the row counts the daily actually reduces.

**Instrument.** `scripts/ar_bench.py` under `torchrun --nproc_per_node=2`, inside a scratch container of the FlashInfer-0.6.18 rc2 image with a FlashInfer main checkout (commit df8b5c1, 2026-09-04) mounted over the package so the `pcie_ipc` kernel JIT-compiles from source; the served image is untouched. Tensors are bf16, rows × 5,120, the daily's hidden size: 10 rows is one stream with the 9-token draft, 80 is eight streams, 160 sixteen, 320 the largest capture size, 2,048 and 8,192 stand in for prefill chunks. Every backend is timed eager (100 calls, median of 5 batches) and inside a CUDA graph (50 captured calls, median of 10 replays); the daily runs decode inside graphs, so the graph column is the served one. Each rank times its own calls and the table reports the group maximum of the per-rank medians, as FlashInfer's own comm benchmark does. Correctness is checked twice per cell: one eager call against an NCCL reference, then one captured call replayed three times into a fixed buffer. The `pcie_ipc` autotune ran on free cards (`pcie-tune-free.json`); a first tune taken while an engine held the cards (21:23 UTC) produced the same kernel choices to within 1%. The run itself took 24 s on free cards between two r183b arms (21:42 UTC). vLLM's `CustomAllreduce` reported `disabled: False`, `full_nvlink: None`.

**Microseconds per all-reduce, both cards, run of 21:42 UTC.** Graph columns are the served path.

| rows | bytes | NCCL eager | NCCL graph | vLLM custom eager | vLLM custom graph | pcie_ipc eager | pcie_ipc graph | pcie_ipc vs custom, graph |
|---|---|---|---|---|---|---|---|---|
| 1 | 10 KB | 11.4 | 13.7 | 6.0 | 4.9 | 7.0 | 1.9 | −61% |
| 8 | 82 KB | 14.5 | 16.7 | 9.3 | 8.1 | 6.9 | 4.8 | −41% |
| 10 (1 stream) | 102 KB | 15.4 | 17.5 | 10.0 | 8.8 | 6.8 | 5.6 | −36% |
| 16 | 164 KB | 20.1 | 22.3 | 12.9 | 11.7 | 8.4 | 8.0 | −32% |
| 20 | 205 KB | 21.9 | 23.7 | 14.3 | 13.1 | 9.9 | 9.6 | −27% |
| 40 | 410 KB | 28.2 | 31.9 | 23.8 | 22.5 | 17.9 | 17.5 | −22% |
| 80 (8 streams) | 819 KB | 43.3 | 46.0 | 41.9 | 49.0 | 33.8 | 33.5 | −32% |
| 160 (16 streams) | 1.6 MB | 77.6 | 80.4 | 77.9 | 86.0 | 65.8 | 65.3 | −24% |
| 320 | 3.3 MB | 146.3 | 149.5 | 150.3 | 160.1 | 129.5 | 129.0 | −19% |
| 2,048 (prefill chunk) | 21 MB | 897.5 | 899.9 | 964.3 | 960.6 | 818.3 | 817.9 | −15% |
| 8,192 | 84 MB | 3,509.8 | 3,508.2 | 3,869.3 | 3,791.5 | 3,282.0 | 3,281.4 | −13% |

All 33 cells matched the NCCL reference exactly (max error 0.0) on the eager call and on all three graph replays.

**The instrument reproduces the engine.** The custom-kernel graph cells at 80 and 160 rows (49.0 and 86.0 µs) match R183's in-engine per-call cost at 8 and 16 streams (49 and 90 µs from the torch profiler). At 1 stream they do not: the profiler saw 19 µs per call and the microbenchmark 8.8, so roughly 10 µs of the in-engine c1 call is something other than the kernel (the profile's inter-kernel gaps are 22% of the step at c1).

**Findings.**

- `pcie_ipc` is faster than the served kernel at every row count: 36% at 10 rows, 32% at 80, 24% at 160, 19% at 320, 13% at 8,192. The edge is largest where the transfer is latency-bound and shrinks toward the bandwidth floor.
- vLLM's custom all-reduce is slower than NCCL in graph mode from 80 rows up (49.0 vs 46.0 µs; 86.0 vs 80.4; 160.1 vs 149.5; 960.6 vs 899.9 at 2,048 rows). It earns its place only below about 40 rows, where it is 2 to 3x faster than NCCL. Raising the 8 MiB custom-all-reduce cap so prefill chunks go through it, which is what [vllm PR #52555](https://github.com/vllm-project/vllm/pull/52555) makes configurable, would cost 7% per prefill all-reduce on this box.
- At 8,192 rows every kernel moves 84 MB in about 3.3 to 3.8 ms, 22 to 26 GB/s per direction, the PCIe Gen5 x8 floor. This is the number behind R183's sentence that the 16-stream all-reduce is closer to bandwidth than to latency.
- Eager and graph timings differ most for `pcie_ipc` at small rows (7.0 eager vs 1.9 graph at 1 row): its eager call carries a host-side launch that the graph elides. NCCL's launch cost dominates its small-row cells in both modes.
- A first run at 21:33 UTC, taken while an r183b engine was serving the dense ruler on the same cards, put NCCL at 149.7 µs for 160 rows against 77.6 free; the custom and `pcie_ipc` cells were unchanged. That run is not in the table.

**What it is worth on the daily.** Multiplying R183's all-reduce share of the decode step by the kernel's relative saving gives the ceiling for a kernel swap: 8 streams 27% × 32% ≈ 8.5%, 16 streams 33% × 24% ≈ 8%, 1 stream between 2.5% (if the 10 µs the profiler saw beyond the kernel stays) and 5.5%. That is the size of the remaining two-card lever, not the 2 to 4x of the FlashInfer PR, whose comparison arm was NCCL.

**Next.** R185 asks whether the kernel can be vendored into the served image without replacing FlashInfer 0.6.16.post3 (the R168 fidelity work depends on that version's attention path), and then routes vLLM's TP all-reduce to it ahead of the custom kernel for bf16 tensors, with every CUDA-graph capture size resolved before capture (a shape first seen inside a capture cannot be resolved, by the kernel's design). The rulers adjudicate the result against the bf16 self-floor, since a different reduction kernel is a different rounding order. Attribution in [THIRD_PARTY.md](../../THIRD_PARTY.md).
