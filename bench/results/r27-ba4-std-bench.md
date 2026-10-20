# r27-ba4-std-bench: 1,589.0 / 1,899.5 wall-clock output tok/s at 16 streams; paired rounds trace the 8-stream gap to L1a to the compile artifact (2026-10-19 to 2026-10-20, results `2026-10-19-r27-ba4-std-bench-2335`, `2026-10-20-025104-r27-ba4-paired-std`, `2026-10-20-040845-r27-ba4-paired-std`)

[← all results](../RESULTS.md)

2026-10-19 23:54 UTC to 2026-10-20 02:37 UTC, raw records in [`2026-10-19-r27-ba4-std-bench-2335/`](2026-10-19-r27-ba4-std-bench-2335/), driver [`scripts/r27-ba4-std-bench.sh`](../../scripts/r27-ba4-std-bench.sh), which is [`scripts/r27-l1a-std-bench.sh`](../../scripts/r27-l1a-std-bench.sh) with the unit name, the asserted launcher md5 and the image name changed. This is R793's standard-benchmark protocol on the served [`scripts/serve-v031-ba4-daily.sh`](../../scripts/serve-v031-ba4-daily.sh), image `vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan`: vLLM v0.31.0, FlashInfer 0.7.0.post1, NVIDIA NVFP4 weights, NVFP4 KV, bf16 linear-attention state, MTP at 3 draft tokens, `pcie_ipc`, batch-sharded sampling, split-KV override, the L1a draft upper-bound plan and the BA4 `in_proj_ba` layer ([r27 BA4](r27-ba4.md)). The boot has 16 sequences, a 262,144-token window, the 14.86 GB pin per card and a 1,391,795-token pool, with 16 GiB CPU and 300 GB capped disk tiers.

The earlier arm is [r27-l1a-std-bench](r27-l1a-std-bench.md), 2026-10-17 09:31 to 12:09 UTC, raw records in [`2026-10-17-r27-l1a-std-bench-0930/`](2026-10-17-r27-l1a-std-bench-0930/), on [`scripts/serve-v031-l1a-daily.sh`](../../scripts/serve-v031-l1a-daily.sh), image `vllm-qwen38:v0310-r27c-A_S-L1a-ubplan`. The two launchers differ in the image's BA4 layer, `VLLM_SM12X_BA_M4=1`, the two admission-report paths and the reports' read-only mount: 48 container environment entries against 45, with the same pool, pin, sequence limit, dtypes, MTP length and samples. Both rounds ran at stock 600 / 575 W power limits with core offset 0 and memory offset +4500, read back at every boot and cell end, and both stopped the GLM container and the same three direct clients before the first boot. Every table below compares these two dated directories with this instrument. At one stream BA4 reads 11.6 and 11.8 % above L1a; at two to sixteen streams every cell reads below it, by up to 6.8 % on ShareGPT V3 and 3.7 % on Spec-Bench ([below](#from-two-streams-on-every-cell-reads-below-l1a)). Two paired rounds on 2026-10-20 trace the 8- and 16-stream throughput gap to the compile artifact each arm loaded, not to the layer; about 5.6 ms of TTFT p50 at 8 streams remains with BA4 on ([paired rounds](#paired-rounds-the-throughput-gap-is-the-compile-artifact-56-ms-of-ttft-at-8-streams-is-ba4)).

## All standard-benchmark cells pass the publication gate

The [summary](2026-10-19-r27-ba4-std-bench-2335/summary.txt) reports `DECISION: PUBLISHABLE (max A/B spread 1.33 % at sharegpt c8; every cell cached <= 1 %)`, with `FLAGS: none`. All 28 boots succeeded on their first attempt with one image identity and engine configuration, the stated pool and no clock or power drift; free VRAM after boot spans 3,063 to 3,819 MiB per card ([boots](2026-10-19-r27-ba4-std-bench-2335/boots.tsv)). Each boot's launcher line, in the boot logs of the results directory (not published here), reads `BA4 96/96 active`. Every cell completed all requests without failures, preemptions, engine errors or foreign requests; server request, prompt-token and generation-token counts equal the client's counts, and both GPU and tier cached prompt tokens are zero ([audit](2026-10-19-r27-ba4-std-bench-2335/audit.txt), [foreign counts](2026-10-19-r27-ba4-std-bench-2335/foreign.tsv)). The production disk tier is kept across boots; the zero-hit counters are measured, rather than inferred from a fresh boot.

## From two streams on, every cell reads below L1a

At one stream, where the BA4 kernel serves every MTP verify step, wall-clock output reads +11.6 % on ShareGPT V3 and +11.8 % on Spec-Bench above L1a, and the per-stream rate +13.5 and +12.5 %. At two to sixteen streams every `in_proj_ba` call carries eight or more rows and runs the original GEMM, and every cell reads below L1a: ShareGPT from −1.8 % at 2 streams to −5.4 % at 8, −6.8 % at 12 and −6.6 % at 16; Spec-Bench from −1.4 % at 2 streams to −3.7 % at 8 and −3.6 % at 16. Time to the first token p50 reads 7.1 to 56.8 % higher in every cell, one stream included. τ is within 0.5 % of L1a in every cell, so draft acceptance does not account for the gap.

The largest A/B spread is 1.33 % in this round and 2.08 % in the L1a round. The ShareGPT deltas from 4 streams on and the Spec-Bench deltas from 6 streams on exceed both. These are two rounds on different days with independent boots, not a paired test: they cannot separate the layer from boot-to-boot or day-to-day variation, and this round does not identify a cause; the paired rounds below do. Decode alone does not show the gap: [r27-ba4-curves](r27-ba4-curves.md) (2026-10-19, results `2026-10-19-233406-r27-ba4-curves`) read −3.0 to +1.9 % against L1a's curves from 2 to 16 streams, −1.2 % on code and +0.2 % on prose at 16, and the order-balanced screens in [r27 BA4](r27-ba4.md) (2026-10-19, results `2026-10-19-r27-audition-screen-070155-3532152`, `2026-10-19-r27-audition-screen-075049-955439`) read the 4- and 8-stream decode cells within noise, with screen 1's code c8 negative in both pairs and screen 2 not reproducing it. The standard benchmark adds what decode alone leaves out: short prompts prefilled beside running decodes, first tokens and request turnover; TTFT, which covers the prefill, reads higher in every cell.

## Paired rounds: the throughput gap is the compile artifact, 5.6 ms of TTFT at 8 streams is BA4

Two rounds on 2026-10-20 paired the layer against itself switched off, driver [`scripts/r27-ba4-paired-std.sh`](../../scripts/r27-ba4-paired-std.sh). Both arms boot the served image and launcher on the experiment port with batch-sharded sampling as served: B1 has BA4 on, B0 sets `BA4=0`, which leaves the layer inactive so the image computes as L1a. The disk tier is wiped before each boot. Each boot runs ShareGPT V3 through the same `vllm bench serve` client and shim, greedy, at 1 stream (100 requests), 8 streams (200) and in round 1 at 16 streams (400), with one seed per cell shared by both arms. The request counts are smaller than this round's 400, so absolute values differ from the tables below. The rules for each round were written into the driver before its data. Raw per-cell records stay in the results directories named here and are not published; the per-boot values below are from each round's audit log.

Round 1, 2026-10-20 02:51 to 03:32 UTC, results `2026-10-20-025104-r27-ba4-paired-std`, booted B1 B0 B0 B1 with the launcher's persistent compile cache. Δ compares the arm medians.

| streams | B1 output tok/s (boots 1, 4) | B0 output tok/s (boots 2, 3) | Δ output | B1 TTFT p50 (ms) | B0 TTFT p50 (ms) | Δ TTFT p50 | B1 TPOT p50 (ms) | B0 TPOT p50 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 236.9, 232.9 | 213.1, 213.4 | +10.1 % | 41.8, 42.3 | 42.7, 42.0 | −0.7 % | 3.90, 4.00 | 4.38, 4.39 |
| 8 | 1,059.2, 1,076.7 | 1,097.5, 1,104.6 | −3.0 % | 116.2, 115.7 | 107.1, 104.6 | +9.5 % | 6.55, 6.41 | 6.44, 6.29 |
| 16 | 1,502.0, 1,520.0 | 1,579.2, 1,565.9 | −3.9 % | 134.1, 131.9 | 121.8, 125.1 | +7.8 % | 9.37, 9.33 | 8.97, 9.08 |

Its rule cleared a cell at an output ratio ≥ 0.98 with TTFT p50 ≤ 1.05×, and confirmed a regression where both boot pairs fell outside both bounds: 1 stream cleared, 8 and 16 streams read as a BA4 regression. A review of the boot logs then found that every B1 boot had loaded one warm ahead-of-time compile artifact and every B0 boot another, compiled cold at boot 2. Round 1 therefore compared two artifacts, one per arm, and could not separate the layer from the artifact draw ([R193](r193-compile-lottery.md) found that compiles of one configuration draw from a discrete set of different artifacts).

Round 2, 2026-10-20 04:08 to 04:57 UTC, results `2026-10-20-040845-r27-ba4-paired-std`, booted B1 B0 B0 B1 B1 B0 at 1 and 8 streams, and every boot compiled cold into its own empty torch.compile, Inductor and Triton caches, so each arm has three independent artifacts. Its rule: if the 8-stream B1/B0 output ratio is ≥ 0.98, or the spread between one arm's draws exceeds the gap between the arms, round 1's 8-stream loss is the artifact draw; if all three B1 draws fall below all three B0 draws, BA4 itself costs 8 streams.

| streams | B1 output tok/s (boots 1, 4, 5) | B0 output tok/s (boots 2, 3, 6) | Δ output | B1 TTFT p50 (ms) | B0 TTFT p50 (ms) | Δ TTFT p50 | B1 TPOT p50 (ms) | B0 TPOT p50 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 242.7, 235.4, 238.5 | 216.8, 214.5, 217.9 | +10.0 % | 42.9, 43.7, 43.1 | 43.8, 42.8, 42.8 | +0.6 % | 3.84, 3.94, 3.95 | 4.36, 4.35, 4.31 |
| 8 | 1,086.3, 1,083.5, 1,069.4 | 1,101.4, 1,087.9, 1,073.2 | −0.4 % | 111.4, 112.4, 112.3 | 108.1, 106.0, 106.7 | +5.3 % | 6.39, 6.46, 6.50 | 6.29, 6.24, 6.36 |

At 8 streams the output ratio is 0.996 and the draws of the two arms interleave: B1 spans 1,069.4 to 1,086.3 tok/s and B0 1,073.2 to 1,101.4, each wider than the 4.4 tok/s between the medians. By the rule, the 8-stream throughput loss of round 1 is the compile artifact, not BA4. The 16-stream cell was not re-run cold; its −3.9 % in round 1 came from the same two warm artifacts. The unpaired comparison above boots the same launcher with its persistent cache, so each of its rounds also measures the artifacts that cache held; the cold draws do not reproduce its 8-stream gap, and the paired rounds did not test 2, 4, 6 or 12 streams.

TTFT is a cost of the layer: at 8 streams every B1 draw's TTFT p50 (111.4 to 112.4 ms) lies above every B0 draw's (106.0 to 108.1 ms), medians 112.3 against 106.7 ms, +5.6 ms or +5.3 %, and TPOT p50 reads 6.46 against 6.29 ms, +2.7 %, every B1 draw (6.39 to 6.50 ms) again above every B0 draw (6.24 to 6.36 ms). At 1 stream TTFT p50 matches within 1 % in both rounds, so the +16.3 % and +7.6 % one-stream TTFT readings of the unpaired comparison are not the layer. The paired rounds do not identify the mechanism of the 8-stream TTFT cost. A cold BA4 compile takes 68.9 to 70.2 s against 26.6 to 26.9 s without the layer, the largest torch.compile time logged per boot.

Net, from the paired rounds: BA4 raises one-stream output by 10.0 to 10.1 %, leaves 8-stream output within 0.4 % on independent compiles, and adds about 5.6 ms of TTFT p50 at 8 streams.

## The two-pass tables compare the same benchmark protocol

The client is `vllm bench serve` v0.30.0 through the [Flash-Next request shim](https://github.com/adrienbrault/qwen3.8-flash-next-2x-rtx5090/blob/main/bench/vllm_bench_tabby.py): streamed OpenAI chat requests, greedy, template-default thinking, `request-rate inf`, closed loop at the stated concurrency. ShareGPT V3 uses 400 conversations with seed 7310 and `min_tokens` forcing each reference reply's length (84,120 output tokens per cell, mean 210.3); Spec-Bench uses all 480 questions forced to 256 tokens each (122,880 output tokens). The prompts and output lengths follow [R793's method](r793-27b-std-bench.md#conditions). Each cell has a fresh engine boot and `c` concurrent, out-of-set 64-token warm-ups, then the counter snapshot; pass A runs every cell before pass B. The outputs are truncated reasoning, not scored answers.

Output tok/s is the client's `output_throughput`, completion tokens divided by total benchmark wall time, including prefill, first-token latency and request turnover. Per stream is `1000 / p50_tpot_ms`, calculated separately for each pass and then averaged; TPOT is `(latency − TTFT) / (output_tokens − 1)` per request and includes waiting behind other requests' prefill. It is not multiplied by concurrency to obtain an aggregate. TTFT and TPOT p50/p99 are the means of the two per-pass quantiles, not quantiles of pooled requests. τ is `generation_tokens / (generation_tokens − accepted_draft_tokens)` from the measured cell counters in `audit.txt`, as in the published `summary.txt`; it differs from the client's `spec_decode_acceptance_length` definition. Each table value is the A/B arithmetic mean; every Δ is `(BA4 / L1a − 1) × 100` before rounding. Higher throughput and lower latency have their usual meanings; Δ τ is a relative percent, not percentage points.

The public JSONs omit only `generated_texts` and `itls`, as in the earlier published rounds. Request lengths, TTFTs, latencies, aggregate statistics and speculative counters remain; `boots.tsv`, `runs.tsv`, `foreign.tsv`, `summary.txt` and `audit.txt` retain the boot identity, accounting and decision evidence. The A/B spread gate applies to output tok/s, not latency quantiles or τ; p99 values describe this sample on these two boots and are not a tail-latency guarantee.


### ShareGPT V3

| streams | L1a output tok/s | BA4 output tok/s | Δ output | L1a per stream | BA4 per stream | Δ per stream | L1a τ | BA4 τ | Δ τ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 221.0 | 246.7 | +11.6 % | 238.1 | 270.3 | +13.5 % | 2.936 | 2.922 | −0.5 % |
| 2 | 407.7 | 400.3 | −1.8 % | 224.4 | 223.2 | −0.5 % | 2.934 | 2.934 | +0.0 % |
| 4 | 751.3 | 728.7 | −3.0 % | 210.5 | 205.4 | −2.4 % | 2.940 | 2.948 | +0.3 % |
| 6 | 985.2 | 960.2 | −2.5 % | 184.4 | 183.4 | −0.5 % | 2.942 | 2.942 | +0.0 % |
| 8 | 1,197.4 | 1,132.7 | −5.4 % | 169.6 | 161.8 | −4.6 % | 2.937 | 2.942 | +0.2 % |
| 12 | 1,491.2 | 1,390.5 | −6.8 % | 140.0 | 131.9 | −5.8 % | 2.939 | 2.927 | −0.4 % |
| 16 | 1,702.0 | 1,589.0 | −6.6 % | 120.3 | 114.3 | −5.0 % | 2.937 | 2.939 | +0.1 % |

Each latency cell is **L1a → BA4 (Δ %)**, in milliseconds; the two pass quantiles are averaged separately.

| streams | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 37.7 → 43.9 (+16.3 %) | 112.1 → 114.5 (+2.1 %) | 4.20 → 3.70 (−11.9 %) | 6.21 → 5.41 (−12.8 %) |
| 2 | 63.8 → 100.0 (+56.8 %) | 132.6 → 148.4 (+11.9 %) | 4.46 → 4.48 (+0.5 %) | 6.53 → 6.80 (+4.2 %) |
| 4 | 99.6 → 110.7 (+11.1 %) | 198.2 → 216.2 (+9.1 %) | 4.75 → 4.87 (+2.5 %) | 9.24 → 10.56 (+14.3 %) |
| 6 | 102.7 → 112.6 (+9.7 %) | 273.4 → 281.3 (+2.9 %) | 5.42 → 5.45 (+0.5 %) | 8.72 → 10.16 (+16.5 %) |
| 8 | 103.1 → 114.8 (+11.3 %) | 307.4 → 308.4 (+0.3 %) | 5.90 → 6.18 (+4.8 %) | 11.28 → 12.15 (+7.7 %) |
| 12 | 107.4 → 121.2 (+12.8 %) | 363.2 → 362.6 (−0.2 %) | 7.14 → 7.58 (+6.2 %) | 14.99 → 20.26 (+35.1 %) |
| 16 | 112.5 → 129.4 (+15.0 %) | 501.7 → 494.1 (−1.5 %) | 8.31 → 8.75 (+5.2 %) | 16.45 → 19.66 (+19.5 %) |

| streams | BA4 output tok/s A/B spread |
| ---: | ---: |
| 1 | 0.42 % |
| 2 | 0.51 % |
| 4 | 1.12 % |
| 6 | 0.48 % |
| 8 | 1.33 % |
| 12 | 0.50 % |
| 16 | 0.42 % |
### Spec-Bench

| streams | L1a output tok/s | BA4 output tok/s | Δ output | L1a per stream | BA4 per stream | Δ per stream | L1a τ | BA4 τ | Δ τ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 247.5 | 276.7 | +11.8 % | 266.8 | 300.1 | +12.5 % | 3.238 | 3.229 | −0.3 % |
| 2 | 457.1 | 450.6 | −1.4 % | 251.0 | 247.9 | −1.3 % | 3.242 | 3.235 | −0.2 % |
| 4 | 844.1 | 830.6 | −1.6 % | 230.4 | 227.5 | −1.3 % | 3.228 | 3.241 | +0.4 % |
| 6 | 1,128.6 | 1,099.1 | −2.6 % | 204.4 | 199.9 | −2.2 % | 3.232 | 3.234 | +0.0 % |
| 8 | 1,389.0 | 1,337.2 | −3.7 % | 188.8 | 181.9 | −3.6 % | 3.231 | 3.238 | +0.2 % |
| 12 | 1,736.3 | 1,677.0 | −3.4 % | 156.7 | 152.5 | −2.7 % | 3.235 | 3.230 | −0.2 % |
| 16 | 1,970.8 | 1,899.5 | −3.6 % | 133.2 | 129.1 | −3.1 % | 3.233 | 3.236 | +0.1 % |

Each latency cell is **L1a → BA4 (Δ %)**, in milliseconds; the two pass quantiles are averaged separately.

| streams | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 35.5 → 38.2 (+7.6 %) | 160.8 → 159.3 (−0.9 %) | 3.75 → 3.33 (−11.1 %) | 5.05 → 4.31 (−14.5 %) |
| 2 | 55.0 → 62.1 (+12.8 %) | 177.9 → 187.3 (+5.3 %) | 3.98 → 4.03 (+1.3 %) | 5.44 → 5.49 (+0.9 %) |
| 4 | 55.9 → 65.5 (+17.1 %) | 205.7 → 195.7 (−4.9 %) | 4.34 → 4.40 (+1.3 %) | 5.80 → 5.93 (+2.1 %) |
| 6 | 64.9 → 69.5 (+7.1 %) | 206.4 → 215.4 (+4.4 %) | 4.89 → 5.00 (+2.2 %) | 6.67 → 6.90 (+3.5 %) |
| 8 | 70.7 → 80.9 (+14.5 %) | 211.5 → 240.6 (+13.7 %) | 5.30 → 5.50 (+3.8 %) | 7.48 → 7.54 (+0.9 %) |
| 12 | 106.3 → 119.9 (+12.9 %) | 235.9 → 249.5 (+5.8 %) | 6.38 → 6.56 (+2.7 %) | 8.94 → 9.46 (+5.7 %) |
| 16 | 116.5 → 131.1 (+12.5 %) | 285.4 → 292.0 (+2.3 %) | 7.50 → 7.75 (+3.2 %) | 10.53 → 11.00 (+4.5 %) |

| streams | BA4 output tok/s A/B spread |
| ---: | ---: |
| 1 | 0.33 % |
| 2 | 0.55 % |
| 4 | 0.31 % |
| 6 | 0.51 % |
| 8 | 0.94 % |
| 12 | 0.20 % |
| 16 | 0.73 % |

## The figure

The dashed lines in [`decode-scaling-16.svg`](../../docs/img/decode-scaling-16.svg), drawn by [`bench/plot.py`](../plot.py), use these BA4 cells; the solid decode-only lines use [r27-ba4-curves](r27-ba4-curves.md), measured on the same launcher 2026-10-19 23:37 to 23:53 UTC, results `2026-10-19-233406-r27-ba4-curves`, with memory offset +4500. Output lengths, text, request turnover, interleaved prefill and boots differ between the solid and dashed lines; their gap is not a measurement of prefill's cost to decode.
