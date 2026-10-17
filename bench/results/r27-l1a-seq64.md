# r27-l1a-seq64: 4,727.2 / 3,914.3 decode aggregate at 64 streams (2026-10-17, results `2026-10-17-r27-l1a-seq64-curves-1226`)

[← all results](../RESULTS.md)

2026-10-17 12:27 to 15:04 UTC, raw records in [`2026-10-17-r27-l1a-seq64-curves-1226/`](2026-10-17-r27-l1a-seq64-curves-1226/), driver [`scripts/r27-l1a-seq64-curves.sh`](../../scripts/r27-l1a-seq64-curves.sh), figure [`decode-scaling-64.svg`](../../docs/img/decode-scaling-64.svg) drawn by [`bench/plot.py`](../plot.py). This is [R794's protocol](r794-27b-seq64.md#conditions) on the L1a image `vllm-qwen38:v0310-r27c-A_S-L1a-ubplan` through [`scripts/serve-v031-l1a-daily.sh`](../../scripts/serve-v031-l1a-daily.sh)'s experiment path. The engine is vLLM v0.31.0 with FlashInfer 0.7.0.post1, NVIDIA NVFP4 weights, NVFP4 KV, bf16 linear-attention state, MTP at 3 draft tokens, `pcie_ipc`, batch-sharded sampling, A_S split-KV and the L1a draft upper-bound plan. The 14.86 GB pin per card and 1,391,795-token pool match the served configuration; this separate configuration raises the sequence limit to 64, caps graph capture at 320, binds port 8029 to loopback and uses a separate disk tier emptied before each boot. The served limit remains 16 sequences.

The earlier arm is [R794](r794-27b-seq64.md), 2026-09-28 20:19 to 23:10 UTC, raw records in [`2026-09-28-r794-27b-seq64-2018/`](2026-09-28-r794-27b-seq64-2018/), using [`scripts/serve-r231-nvidia-daily.sh`](../../scripts/serve-r231-nvidia-daily.sh)'s 64-sequence experiment path on vLLM v0.29.0rc2 and FlashInfer 0.6.16.post3. Its accepted [R794b ShareGPT c16 re-run](r794-27b-seq64.md#sharegpt-at-16-streams-comes-from-a-re-run-of-that-pair), 2026-09-28 23:13 to 23:22 UTC, raw records in [`2026-09-28-r794b-27b-c16-2312/`](2026-09-28-r794b-27b-c16-2312/), supplies that historical cell here. Both arms use the same checkpoint, pin, pool, sequence limit, graph cap, MTP length, dtypes, benchmark samples and seed scheme, stock 600 / 575 W power limits, core offset 0 and memory offset +4500, checked at each boot and cell end. Every table below compares those dated directories with the stated instrument.

## All decode shapes and standard-benchmark cells pass

The [audit](2026-10-17-r27-l1a-seq64-curves-1226/audit.txt) reports `PHASE 1: OK` and a final `PUBLISHABLE` decision with `14/14 PUBLISHABLE`; the maximum A/B output-rate spread is 1.48 % at ShareGPT c8 ([summary](2026-10-17-r27-l1a-seq64-curves-1226/summary.txt), [cell verdicts](2026-10-17-r27-l1a-seq64-curves-1226/cells-verdict.tsv)). All 29 boots succeeded on their first attempt, one for decode alone and 28 for the benchmark, with one image, one configuration, the stated pool and the same two compiled artifacts. Free VRAM after boot spans 2,509 to 3,193 MiB per card ([boots](2026-10-17-r27-l1a-seq64-curves-1226/boots.tsv)). Every decode shape has three good runs. All phases report zero preemptions and engine errors; every benchmark cell has zero failed requests, foreign requests and GPU/tier cached prompt tokens, with request, prompt and generation counts matching the client.

## Decode aggregate rises through 64 streams

[`scripts/decode_ss.py`](../../scripts/decode_ss.py) starts every stream together, forces 1,024 tokens per stream at temperature 0, and measures only samples where all `c` streams were decoding. One code and one prose prompt are used with a fresh prefix per invocation, three runs per shape on one boot; the values below are the raw summary medians from each arm's `decode-code.jsonl` / `decode-prose.jsonl`. Aggregate is the measured sum of stream rates and per stream is the probe's separate summary. Acceptance is accepted draft tokens / drafted tokens; Δ acceptance is in percentage points (pp).

### Code

| streams | R794 aggregate | L1a aggregate | Δ aggregate | R794 per stream | L1a per stream | Δ per stream | R794 acceptance | L1a acceptance | Δ acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 221.7 | 234.0 | +5.5 % | 221.7 | 234.0 | +5.5 % | 0.663 | 0.661 | −0.20 pp |
| 2 | 429.0 | 499.0 | +16.3 % | 214.5 | 249.5 | +16.3 % | 0.638 | 0.730 | +9.20 pp |
| 4 | 878.6 | 938.4 | +6.8 % | 219.7 | 234.6 | +6.8 % | 0.657 | 0.671 | +1.40 pp |
| 8 | 1,618.0 | 1,618.2 | +0.0 % | 202.2 | 202.3 | +0.0 % | 0.655 | 0.621 | −3.40 pp |
| 16 | 2,698.9 | 2,664.2 | −1.3 % | 168.7 | 166.5 | −1.3 % | 0.658 | 0.633 | −2.50 pp |
| 24 | 3,337.2 | 3,226.9 | −3.3 % | 139.0 | 134.5 | −3.2 % | 0.643 | 0.628 | −1.50 pp |
| 32 | 3,838.1 | 3,780.6 | −1.5 % | 119.9 | 118.1 | −1.5 % | 0.636 | 0.654 | +1.80 pp |
| 48 | 4,361.9 | 4,156.6 | −4.7 % | 90.9 | 86.6 | −4.7 % | 0.653 | 0.641 | −1.20 pp |
| 64 | 4,807.4 | 4,727.2 | −1.7 % | 75.1 | 73.9 | −1.6 % | 0.655 | 0.657 | +0.20 pp |

### Prose

| streams | R794 aggregate | L1a aggregate | Δ aggregate | R794 per stream | L1a per stream | Δ per stream | R794 acceptance | L1a acceptance | Δ acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 178.6 | 195.4 | +9.4 % | 178.6 | 195.4 | +9.4 % | 0.467 | 0.497 | +3.00 pp |
| 2 | 361.3 | 390.6 | +8.1 % | 180.7 | 195.3 | +8.1 % | 0.477 | 0.501 | +2.40 pp |
| 4 | 745.7 | 756.4 | +1.4 % | 186.4 | 189.1 | +1.4 % | 0.477 | 0.476 | −0.10 pp |
| 8 | 1,320.7 | 1,392.3 | +5.4 % | 165.1 | 174.0 | +5.4 % | 0.475 | 0.489 | +1.40 pp |
| 16 | 2,203.0 | 2,223.1 | +0.9 % | 137.7 | 138.9 | +0.9 % | 0.478 | 0.472 | −0.60 pp |
| 24 | 2,790.7 | 2,759.3 | −1.1 % | 116.3 | 115.0 | −1.1 % | 0.483 | 0.481 | −0.20 pp |
| 32 | 3,190.5 | 3,123.2 | −2.1 % | 99.7 | 97.6 | −2.1 % | 0.476 | 0.483 | +0.70 pp |
| 48 | 3,646.0 | 3,500.9 | −4.0 % | 76.0 | 72.9 | −4.1 % | 0.492 | 0.487 | −0.50 pp |
| 64 | 3,961.0 | 3,914.3 | −1.2 % | 61.9 | 61.2 | −1.1 % | 0.481 | 0.487 | +0.60 pp |

The code c1 runs span 189.2 to 250.4 aggregate t/s around a 234.0 median (26.2 % of the median); those medians describe one boot, not a precision bound on the configuration. At c64 every run contains a window with all 64 requests decoding: 11.6 to 12.1 s for code and 14.6 to 15.1 s for prose. The benchmark logs also reach 64 running requests, with a peak KV-cache gauge of 81.0 % across its cells; no request is preempted. This demonstrates 64 simultaneous short requests at this pin, not 64 long contexts.

## The two-pass standard benchmark uses a fresh boot per cell

The client is `vllm bench serve` v0.30.0 through the [Flash-Next request shim](https://github.com/adrienbrault/qwen3.8-flash-next-2x-rtx5090/blob/main/bench/vllm_bench_tabby.py): streamed OpenAI chat requests, greedy, template-default thinking, `request-rate inf`, closed loop at the stated concurrency. ShareGPT V3 uses 400 conversations with seed 7310 and `min_tokens` forcing each reference reply's length (84,120 output tokens per cell, mean 210.3); Spec-Bench uses all 480 questions forced to 256 tokens each (122,880 output tokens). The prompts and output lengths follow [R793's method](r793-27b-std-bench.md#conditions). Each cell has a fresh engine boot and `c` concurrent, out-of-set 64-token warm-ups, then the counter snapshot; pass A runs every cell before pass B. The outputs are truncated reasoning, not scored answers.

Output tok/s is the client's `output_throughput`, completion tokens divided by total benchmark wall time, including prefill, first-token latency and request turnover. Per stream is `1000 / p50_tpot_ms`, calculated separately for each pass and then averaged; TPOT is `(latency − TTFT) / (output_tokens − 1)` per request and includes waiting behind other requests' prefill. It is not multiplied by concurrency to obtain an aggregate. TTFT and TPOT p50/p99 are the means of the two per-pass quantiles, not quantiles of pooled requests. τ follows the published `summary.txt`: `generation_tokens / (generation_tokens − accepted_draft_tokens)` from the measured cell counters, reconstructed from the JSON token counts and checked against `audit.txt`; it differs from the client's `spec_decode_acceptance_length` definition. Each table value is the A/B arithmetic mean; every Δ is `(L1a / earlier − 1) × 100` before rounding. Higher throughput and lower latency have their usual meanings; Δ τ is a relative percent, not percentage points.

The public JSONs omit only `generated_texts` and `itls`, as in the earlier published rounds. Request lengths, TTFTs, latencies, aggregate statistics and speculative counters remain; `boots.tsv`, `runs.tsv`, `foreign.tsv`, `summary.txt` and `audit.txt` retain the boot identity, accounting and decision evidence. The A/B spread gate applies to output tok/s, not latency quantiles or τ; p99 values describe this sample on these two boots and are not a tail-latency guarantee.

### ShareGPT V3

| streams | R794 output tok/s | L1a output tok/s | Δ output | R794 per stream | L1a per stream | Δ per stream | R794 τ | L1a τ | Δ τ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 206.9 | 224.3 | +8.4 % | 220.9 | 238.5 | +8.0 % | 2.928 | 2.927 | −0.0 % |
| 4 | 703.5 | 769.6 | +9.4 % | 202.7 | 212.8 | +5.0 % | 2.925 | 2.933 | +0.3 % |
| 8 | 1,069.4 | 1,244.7 | +16.4 % | 152.9 | 174.8 | +14.3 % | 2.939 | 2.940 | +0.0 % |
| 16 | 1,505.9 | 1,802.0 | +19.7 % | 106.9 | 126.8 | +18.6 % | 2.941 | 2.927 | −0.5 % |
| 32 | 1,845.8 | 2,240.1 | +21.4 % | 66.2 | 80.8 | +22.0 % | 2.935 | 2.934 | −0.0 % |
| 48 | 1,992.8 | 2,410.9 | +21.0 % | 47.6 | 58.2 | +22.3 % | 2.926 | 2.942 | +0.6 % |
| 64 | 2,009.8 | 2,507.7 | +24.8 % | 34.8 | 44.9 | +28.9 % | 2.932 | 2.944 | +0.4 % |

Each latency cell is **R794 → L1a (Δ %)**, in milliseconds; the two pass quantiles are averaged separately.

| streams | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 43.2 → 36.4 (−15.9 %) | 174.2 → 112.9 (−35.2 %) | 4.53 → 4.19 (−7.4 %) | 6.50 → 6.08 (−6.5 %) |
| 4 | 67.7 → 60.6 (−10.5 %) | 291.9 → 179.4 (−38.5 %) | 4.93 → 4.70 (−4.8 %) | 11.63 → 9.62 (−17.3 %) |
| 8 | 74.2 → 71.9 (−3.2 %) | 407.1 → 309.3 (−24.0 %) | 6.54 → 5.72 (−12.5 %) | 14.66 → 10.90 (−25.7 %) |
| 16 | 135.4 → 110.2 (−18.6 %) | 588.5 → 500.7 (−14.9 %) | 9.36 → 7.89 (−15.7 %) | 24.47 → 16.17 (−33.9 %) |
| 32 | 201.4 → 143.5 (−28.8 %) | 985.8 → 912.7 (−7.4 %) | 15.10 → 12.37 (−18.0 %) | 39.05 → 25.22 (−35.4 %) |
| 48 | 228.8 → 188.5 (−17.6 %) | 1437.0 → 1289.9 (−10.2 %) | 21.00 → 17.18 (−18.2 %) | 50.19 → 40.62 (−19.1 %) |
| 64 | 335.3 → 232.4 (−30.7 %) | 1721.3 → 1579.6 (−8.2 %) | 28.71 → 22.27 (−22.4 %) | 68.33 → 56.64 (−17.1 %) |

| streams | L1a output tok/s A/B spread |
| ---: | ---: |
| 1 | 0.14 % |
| 4 | 0.09 % |
| 8 | 1.48 % |
| 16 | 0.68 % |
| 32 | 0.27 % |
| 48 | 0.24 % |
| 64 | 0.66 % |

### Spec-Bench

| streams | R794 output tok/s | L1a output tok/s | Δ output | R794 per stream | L1a per stream | Δ per stream | R794 τ | L1a τ | Δ τ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 230.3 | 248.6 | +8.0 % | 250.9 | 267.3 | +6.5 % | 3.231 | 3.232 | +0.0 % |
| 4 | 790.9 | 854.6 | +8.0 % | 218.9 | 232.3 | +6.1 % | 3.224 | 3.235 | +0.3 % |
| 8 | 1,230.7 | 1,390.0 | +12.9 % | 167.7 | 189.1 | +12.8 % | 3.230 | 3.232 | +0.1 % |
| 16 | 1,769.4 | 2,061.9 | +16.5 % | 120.3 | 139.5 | +16.0 % | 3.231 | 3.230 | −0.0 % |
| 32 | 2,264.6 | 2,628.0 | +16.0 % | 76.3 | 88.9 | +16.6 % | 3.221 | 3.231 | +0.3 % |
| 48 | 2,521.4 | 2,877.3 | +14.1 % | 57.3 | 66.2 | +15.5 % | 3.232 | 3.236 | +0.1 % |
| 64 | 2,570.8 | 2,991.7 | +16.4 % | 43.8 | 51.6 | +17.8 % | 3.232 | 3.229 | −0.1 % |

Each latency cell is **R794 → L1a (Δ %)**, in milliseconds; the two pass quantiles are averaged separately.

| streams | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 42.8 → 35.1 (−18.0 %) | 210.0 → 158.3 (−24.6 %) | 3.99 → 3.74 (−6.2 %) | 5.26 → 4.96 (−5.6 %) |
| 4 | 63.7 → 55.8 (−12.4 %) | 240.1 → 197.8 (−17.6 %) | 4.57 → 4.30 (−5.8 %) | 6.39 → 5.73 (−10.4 %) |
| 8 | 74.0 → 71.1 (−4.0 %) | 275.6 → 216.0 (−21.6 %) | 5.96 → 5.29 (−11.4 %) | 8.52 → 7.24 (−15.1 %) |
| 16 | 104.4 → 101.0 (−3.3 %) | 390.8 → 284.4 (−27.2 %) | 8.31 → 7.17 (−13.8 %) | 12.20 → 10.08 (−17.4 %) |
| 32 | 203.3 → 155.6 (−23.5 %) | 896.6 → 835.4 (−6.8 %) | 13.11 → 11.24 (−14.2 %) | 18.57 → 15.34 (−17.4 %) |
| 48 | 244.8 → 201.6 (−17.6 %) | 1524.8 → 1364.0 (−10.5 %) | 17.44 → 15.10 (−13.4 %) | 25.61 → 21.95 (−14.3 %) |
| 64 | 362.3 → 273.6 (−24.5 %) | 2045.7 → 2017.3 (−1.4 %) | 22.85 → 19.40 (−15.1 %) | 35.05 → 28.53 (−18.6 %) |

| streams | L1a output tok/s A/B spread |
| ---: | ---: |
| 1 | 0.26 % |
| 4 | 0.23 % |
| 8 | 0.69 % |
| 16 | 0.32 % |
| 32 | 0.06 % |
| 48 | 0.45 % |
| 64 | 0.14 % |

## The comparison spans v0.29 to A_S to L1a

At 64 streams, ShareGPT and Spec-Bench wall-clock output rates read +24.8 % and +16.4 % above the published R794 values, while decode-only aggregate reads −1.7 % for code and −1.2 % for prose. These are different days and independently booted rounds: the change between them is v0.29.0rc2 / FlashInfer 0.6.16 → v0.31.0 / FlashInfer 0.7 with A_S → L1a, not one lever. The mixed-workload increase and decode-only decrease cannot be attributed separately to L1a, FlashInfer, the runner, scheduler or kernels. The order-balanced [L1a screen](r27-l1a.md), 2026-10-17, results `2026-10-17-r27-audition-screen-022747-4123606`, is the controlled layer comparison.

The [served 16-sequence measurements](r27-l1a-std-bench.md), 2026-10-17, results `2026-10-17-r27-l1a-std-bench-0930`, and this 64-sequence shape are not an interleaved A/B: sequence limit, graph capture, compiled artifact, port, disk tier and session differ. They do not isolate the effect of raising the sequence limit. The solid/dashed gap also mixes different text, output lengths, turnover and prefill; it does not isolate a prefill penalty. At 64 streams the finite samples contain 6.25 ShareGPT waves and 7.5 Spec-Bench waves; the variable-length ShareGPT tail can reduce concurrency near the end. The output-rate curve does not establish a server throughput ceiling or long-context capacity, fidelity or answer quality.
