# r27-l1a-std-bench: 1,702.0 / 1,970.8 wall-clock output tok/s at 16 streams (2026-10-17, results `2026-10-17-r27-l1a-std-bench-0930`)

[← all results](../RESULTS.md)

2026-10-17 09:31 to 12:09 UTC, raw records in [`2026-10-17-r27-l1a-std-bench-0930/`](2026-10-17-r27-l1a-std-bench-0930/), driver [`scripts/r27-l1a-std-bench.sh`](../../scripts/r27-l1a-std-bench.sh). This is R793's standard-benchmark protocol on the served [`scripts/serve-v031-l1a-daily.sh`](../../scripts/serve-v031-l1a-daily.sh), image `vllm-qwen38:v0310-r27c-A_S-L1a-ubplan`: vLLM v0.31.0, FlashInfer 0.7.0.post1, NVIDIA NVFP4 weights, NVFP4 KV, bf16 linear-attention state, MTP at 3 draft tokens, `pcie_ipc`, batch-sharded sampling, split-KV override and the L1a draft upper-bound plan. The boot has 16 sequences, a 262,144-token window, the 14.86 GB pin per card and a 1,391,795-token pool, with 16 GiB CPU and 300 GB capped disk tiers.

The earlier arm is [R793](r793-27b-std-bench.md), 2026-09-28 16:09 to 19:04 UTC, raw records in [`2026-09-28-r793-27b-std-bench-1608/`](2026-09-28-r793-27b-std-bench-1608/), on [`scripts/serve-r231-nvidia-daily.sh`](../../scripts/serve-r231-nvidia-daily.sh), vLLM v0.29.0rc2 and FlashInfer 0.6.16.post3. Its published ShareGPT c6 cell comes from [R793b](r793-27b-std-bench.md#sharegpt-at-6-streams-comes-from-a-re-run-of-that-pair), 2026-09-28 19:05 to 19:16 UTC, raw records in [`2026-09-28-r793b-27b-c6-1905/`](2026-09-28-r793b-27b-c6-1905/); that accepted re-run is the baseline here. Both arms use the same checkpoint, pin, pool, sequence limit, dtypes, MTP length and samples, stock 600 / 575 W power limits, core offset 0 and memory offset +4500, read back at every boot and cell end. Every table below compares these dated directories with this instrument.

## All standard-benchmark cells pass the publication gate

The [summary](2026-10-17-r27-l1a-std-bench-0930/summary.txt) reports `DECISION: PUBLISHABLE (max A/B spread 2.08 % at sharegpt c1; every cell cached <= 1 %)`, with `FLAGS: none`. All 28 boots succeeded on their first attempt with one image identity and engine configuration, the stated pool and no clock or power drift; free VRAM after boot spans 3,107 to 3,967 MiB per card ([boots](2026-10-17-r27-l1a-std-bench-0930/boots.tsv)). Every cell completed all requests without failures, preemptions, engine errors or foreign requests; server request, prompt-token and generation-token counts equal the client's counts, and both GPU and tier cached prompt tokens are zero ([audit](2026-10-17-r27-l1a-std-bench-0930/audit.txt), [foreign counts](2026-10-17-r27-l1a-std-bench-0930/foreign.tsv)). The production disk tier is kept across boots; the zero-hit counters are measured, rather than inferred from a fresh boot.

## The two-pass tables compare the same benchmark protocol

The client is `vllm bench serve` v0.30.0 through the [Flash-Next request shim](https://github.com/adrienbrault/qwen3.8-flash-next-2x-rtx5090/blob/main/bench/vllm_bench_tabby.py): streamed OpenAI chat requests, greedy, template-default thinking, `request-rate inf`, closed loop at the stated concurrency. ShareGPT V3 uses 400 conversations with seed 7310 and `min_tokens` forcing each reference reply's length (84,120 output tokens per cell, mean 210.3); Spec-Bench uses all 480 questions forced to 256 tokens each (122,880 output tokens). The prompts and output lengths follow [R793's method](r793-27b-std-bench.md#conditions). Each cell has a fresh engine boot and `c` concurrent, out-of-set 64-token warm-ups, then the counter snapshot; pass A runs every cell before pass B. The outputs are truncated reasoning, not scored answers.

Output tok/s is the client's `output_throughput`, completion tokens divided by total benchmark wall time, including prefill, first-token latency and request turnover. Per stream is `1000 / p50_tpot_ms`, calculated separately for each pass and then averaged; TPOT is `(latency − TTFT) / (output_tokens − 1)` per request and includes waiting behind other requests' prefill. It is not multiplied by concurrency to obtain an aggregate. TTFT and TPOT p50/p99 are the means of the two per-pass quantiles, not quantiles of pooled requests. τ follows the published `summary.txt`: `generation_tokens / (generation_tokens − accepted_draft_tokens)` from the measured cell counters, reconstructed from the JSON token counts and checked against `audit.txt`; it differs from the client's `spec_decode_acceptance_length` definition. Each table value is the A/B arithmetic mean; every Δ is `(L1a / earlier − 1) × 100` before rounding. Higher throughput and lower latency have their usual meanings; Δ τ is a relative percent, not percentage points.

The public JSONs omit only `generated_texts` and `itls`, as in the earlier published rounds. Request lengths, TTFTs, latencies, aggregate statistics and speculative counters remain; `boots.tsv`, `runs.tsv`, `foreign.tsv`, `summary.txt` and `audit.txt` retain the boot identity, accounting and decision evidence. The A/B spread gate applies to output tok/s, not latency quantiles or τ; p99 values describe this sample on these two boots and are not a tail-latency guarantee.

### ShareGPT V3

| streams | R793 output tok/s | L1a output tok/s | Δ output | R793 per stream | L1a per stream | Δ per stream | R793 τ | L1a τ | Δ τ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 204.8 | 221.0 | +7.9 % | 224.4 | 238.1 | +6.1 % | 2.943 | 2.936 | −0.3 % |
| 2 | 364.5 | 407.7 | +11.9 % | 211.8 | 224.4 | +5.9 % | 2.933 | 2.934 | +0.0 % |
| 4 | 664.2 | 751.3 | +13.1 % | 196.4 | 210.5 | +7.2 % | 2.929 | 2.940 | +0.4 % |
| 6 | 832.7 | 985.2 | +18.3 % | 163.4 | 184.4 | +12.9 % | 2.921 | 2.942 | +0.7 % |
| 8 | 990.2 | 1,197.4 | +20.9 % | 142.8 | 169.6 | +18.8 % | 2.928 | 2.937 | +0.3 % |
| 12 | 1,208.3 | 1,491.2 | +23.4 % | 116.9 | 140.0 | +19.8 % | 2.927 | 2.939 | +0.4 % |
| 16 | 1,330.1 | 1,702.0 | +28.0 % | 97.4 | 120.3 | +23.5 % | 2.930 | 2.937 | +0.2 % |

Each latency cell is **R793 → L1a (Δ %)**, in milliseconds; the two pass quantiles are averaged separately.

| streams | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 44.2 → 37.7 (−14.7 %) | 166.6 → 112.1 (−32.7 %) | 4.46 → 4.20 (−5.8 %) | 6.52 → 6.21 (−4.8 %) |
| 2 | 105.8 → 63.8 (−39.7 %) | 226.0 → 132.6 (−41.3 %) | 4.72 → 4.46 (−5.6 %) | 7.46 → 6.53 (−12.4 %) |
| 4 | 115.2 → 99.6 (−13.6 %) | 318.7 → 198.2 (−37.8 %) | 5.09 → 4.75 (−6.7 %) | 14.85 → 9.24 (−37.8 %) |
| 6 | 137.9 → 102.7 (−25.5 %) | 365.2 → 273.4 (−25.1 %) | 6.12 → 5.42 (−11.4 %) | 14.87 → 8.72 (−41.4 %) |
| 8 | 167.1 → 103.1 (−38.3 %) | 407.6 → 307.4 (−24.6 %) | 7.00 → 5.90 (−15.8 %) | 17.98 → 11.28 (−37.3 %) |
| 12 | 170.7 → 107.4 (−37.1 %) | 452.8 → 363.2 (−19.8 %) | 8.56 → 7.14 (−16.5 %) | 20.84 → 14.99 (−28.1 %) |
| 16 | 175.9 → 112.5 (−36.0 %) | 592.4 → 501.7 (−15.3 %) | 10.27 → 8.31 (−19.0 %) | 29.89 → 16.45 (−45.0 %) |

| streams | L1a output tok/s A/B spread |
| ---: | ---: |
| 1 | 2.08 % |
| 2 | 0.50 % |
| 4 | 0.32 % |
| 6 | 0.58 % |
| 8 | 1.93 % |
| 12 | 0.38 % |
| 16 | 0.29 % |

### Spec-Bench

| streams | R793 output tok/s | L1a output tok/s | Δ output | R793 per stream | L1a per stream | Δ per stream | R793 τ | L1a τ | Δ τ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 228.8 | 247.5 | +8.2 % | 250.3 | 266.8 | +6.6 % | 3.240 | 3.238 | −0.1 % |
| 2 | 414.7 | 457.1 | +10.2 % | 232.5 | 251.0 | +8.0 % | 3.226 | 3.242 | +0.5 % |
| 4 | 775.7 | 844.1 | +8.8 % | 214.8 | 230.4 | +7.3 % | 3.226 | 3.228 | +0.1 % |
| 6 | 993.7 | 1,128.6 | +13.6 % | 182.6 | 204.4 | +12.0 % | 3.229 | 3.232 | +0.1 % |
| 8 | 1,173.3 | 1,389.0 | +18.4 % | 160.8 | 188.8 | +17.4 % | 3.224 | 3.231 | +0.2 % |
| 12 | 1,458.3 | 1,736.3 | +19.1 % | 132.7 | 156.7 | +18.0 % | 3.227 | 3.235 | +0.2 % |
| 16 | 1,627.5 | 1,970.8 | +21.1 % | 111.5 | 133.2 | +19.5 % | 3.229 | 3.233 | +0.1 % |

Each latency cell is **R793 → L1a (Δ %)**, in milliseconds; the two pass quantiles are averaged separately.

| streams | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 42.8 → 35.5 (−17.1 %) | 204.2 → 160.8 (−21.3 %) | 3.99 → 3.75 (−6.2 %) | 5.34 → 5.05 (−5.4 %) |
| 2 | 64.7 → 55.0 (−14.9 %) | 238.6 → 177.9 (−25.5 %) | 4.30 → 3.98 (−7.4 %) | 5.88 → 5.44 (−7.5 %) |
| 4 | 64.1 → 55.9 (−12.7 %) | 284.8 → 205.7 (−27.8 %) | 4.66 → 4.34 (−6.8 %) | 6.58 → 5.80 (−11.7 %) |
| 6 | 77.4 → 64.9 (−16.1 %) | 301.4 → 206.4 (−31.5 %) | 5.48 → 4.89 (−10.7 %) | 7.93 → 6.67 (−16.0 %) |
| 8 | 90.5 → 70.7 (−22.0 %) | 302.0 → 211.5 (−30.0 %) | 6.22 → 5.30 (−14.8 %) | 8.84 → 7.48 (−15.5 %) |
| 12 | 158.5 → 106.3 (−33.0 %) | 346.0 → 235.9 (−31.8 %) | 7.54 → 6.38 (−15.3 %) | 11.02 → 8.94 (−18.9 %) |
| 16 | 175.8 → 116.5 (−33.7 %) | 361.1 → 285.4 (−21.0 %) | 8.97 → 7.50 (−16.3 %) | 13.43 → 10.53 (−21.6 %) |

| streams | L1a output tok/s A/B spread |
| ---: | ---: |
| 1 | 0.37 % |
| 2 | 0.57 % |
| 4 | 0.25 % |
| 6 | 0.27 % |
| 8 | 0.38 % |
| 12 | 0.07 % |
| 16 | 0.16 % |

## The difference spans the engine upgrade and the L1a layer

At 16 streams, ShareGPT reads +28.0 % and Spec-Bench +21.1 % above the published R793 wall-clock output rates. These are measurements on different days with independent boots, not a paired test of one lever: the path between them is v0.29.0rc2 / FlashInfer 0.6.16 → v0.31.0 / FlashInfer 0.7 with the A_S split-KV override → L1a. They cannot assign the difference to L1a, the runner, kernels, scheduler or FlashInfer separately, establish fidelity or answer quality, or predict long-context rates. The [order-balanced L1a screen](r27-l1a.md), 2026-10-17, results `2026-10-17-r27-audition-screen-022747-4123606`, is the controlled comparison of the layer against A_S.

The dashed lines in [`decode-scaling-16.svg`](../../docs/img/decode-scaling-16.svg), drawn by [`bench/plot.py`](../plot.py), use these L1a cells; solid decode-only lines use [r27-l1a-curves](r27-l1a-curves.md), measured 2026-10-17 07:12 to 07:30 UTC, results `2026-10-17-r27-l1a-curves`. Output lengths, text, request turnover, interleaved prefill and boots differ between the solid and dashed lines, and the decode-only round did not record its memory clock offset. Their gap is not a measurement of prefill's cost to decode.
