# R793: `vllm bench serve` on the served configuration reads 1,330 (ShareGPT V3) and 1,628 (Spec-Bench) output tok/s at 16 streams, both passes within 1.04 % of each other in every cell

2026-09-28 16:09 to 19:04 UTC, results `2026-09-28-r793-27b-std-bench-1608` (raw records in [`2026-09-28-r793-27b-std-bench-1608/`](2026-09-28-r793-27b-std-bench-1608/)); ShareGPT at 6 streams from the re-run on 2026-09-28 19:05 to 19:16 UTC, results `2026-09-28-r793b-27b-c6-1905` (raw records in [`2026-09-28-r793b-27b-c6-1905/`](2026-09-28-r793b-27b-c6-1905/)); driver [`scripts/r793-27b-std-bench.sh`](../../scripts/r793-27b-std-bench.sh); dashed lines of `docs/img/decode-scaling-16.svg` drawn by [`bench/plot.py`](../plot.py). The raw records are the client's per-cell JSONs without `generated_texts` and `itls`, plus `summary.txt`, `boots.tsv`, `runs.tsv`, `foreign.tsv` and `audit.txt`.

The R793 records supplied the earlier standard-benchmark figure. Since 2026-10-17, `decode-scaling-16.svg` uses the L1a re-measurement, results `2026-10-17-r27-l1a-std-bench-0930`; [r27-l1a-std-bench](r27-l1a-std-bench.md) places these published R793/R793b values beside the L1a values and their deltas. The measurements below remain those of the dated v0.29.0rc2 configuration.

## Conditions

- **Configuration:** the served launcher (R231/R234: nvidia/Qwen3.8-27B-NVFP4, NVFP4 KV at the 14.86 GB pin, MTP at 3 draft tokens, `pcie_ipc` all-reduce, batch-sharded sampling, 16 sequences). Image `vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash-mtppcie-mtpcache-eagleshift`, port 8020.
- **Boots:** a fresh boot for every cell, 28 in R793 and 2 in R793b. All 30 had the same image id, engine arguments and environment, a 1,391,795-token pool (R675's), stock power limits of 600 / 575 W, core clock offset 0 and memory clock offset +4500, read back at boot and after each cell.
- **Client:** vLLM's serving benchmark, `vllm bench serve` v0.30.0, through the request shim [`bench/vllm_bench_tabby.py`](https://github.com/adrienbrault/qwen3.8-flash-next-2x-rtx5090/blob/main/bench/vllm_bench_tabby.py) of the Flash-Next repository. The shim forces length with `min_tokens` and makes Spec-Bench prompts templated once.
- **Requests:** closed loop at `c` concurrent requests, `c` = 1, 2, 4, 6, 8, 12, 16; greedy, streamed, thinking on (the template default).
- **Samples, identical at every concurrency:**
  - ShareGPT V3: 400 conversations, seed 7310. Each output is forced to the reference reply's length (mean 210 tokens, 4 to 1,439).
  - Spec-Bench: all 480 questions, each output forced to 256 tokens.
  - These are the prompts and output lengths of the Flash-Next R787d round.
- **Outputs:** truncated reasoning, not answers.
- **Inputs:** short, mean 230 / 280 tokens by the client's count, maximum 1,028 / 1,498 (ShareGPT / Spec-Bench).
- **Passes:** A ran all 14 cells, then B ran them again; a cell is the mean of the two.
- **Metrics:**
  - Output tok/s is all completion tokens over the time from the first request's start to the last completion, prefill, time to first token and turnover between requests included.
  - Per stream is 1000 / TPOT p50. TPOT is (latency − time to first token) / (output tokens − 1) per request, so it includes the steps a request waits while other requests' prefill chunks run.

## Results

**ShareGPT V3**

| streams | output tok/s | pass A / B | A/B spread | per stream | TPOT p50 A / B (ms) | TTFT p50 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 204.8 | 205.7 / 203.9 | 0.89 % | 224 | 4.46 / 4.45 | 44 |
| 2 | 364.5 | 362.8 / 366.1 | 0.90 % | 212 | 4.72 / 4.72 | 99-112 |
| 4 | 664.2 | 665.9 / 662.5 | 0.51 % | 196 | 5.06 / 5.12 | 114-116 |
| 6 (R793b) | 832.7 | 829.8 / 835.7 | 0.71 % | 163 | 6.16 / 6.08 | 119-157 |
| 8 | 990.2 | 993.9 / 986.4 | 0.76 % | 143 | 7.03 / 6.98 | 167 |
| 12 | 1,208.3 | 1,207.2 / 1,209.4 | 0.18 % | 117 | 8.64 / 8.48 | 170-171 |
| 16 | 1,330.1 | 1,334.2 / 1,326.1 | 0.61 % | 97 | 10.25 / 10.29 | 176 |

**Spec-Bench**

| streams | output tok/s | pass A / B | A/B spread | per stream | TPOT p50 A / B (ms) | TTFT p50 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 228.8 | 228.7 / 228.9 | 0.05 % | 250 | 4.00 / 3.99 | 42-43 |
| 2 | 414.7 | 413.3 / 416.0 | 0.65 % | 233 | 4.31 / 4.29 | 65 |
| 4 | 775.7 | 775.2 / 776.2 | 0.13 % | 215 | 4.64 / 4.68 | 64 |
| 6 | 993.7 | 998.8 / 988.5 | 1.04 % | 183 | 5.44 / 5.52 | 68-87 |
| 8 | 1,173.3 | 1,174.8 / 1,171.8 | 0.25 % | 161 | 6.21 / 6.23 | 88-93 |
| 12 | 1,458.3 | 1,465.5 / 1,451.1 | 0.98 % | 133 | 7.48 / 7.60 | 146-171 |
| 16 | 1,627.5 | 1,625.5 / 1,629.4 | 0.24 % | 111 | 9.02 / 8.92 | 176 |

- **Speculation:** MTP accepted 2.92 to 2.96 tokens per verify step on ShareGPT and 3.22 to 3.24 on Spec-Bench in every cell, from the server's counters.
- **TTFT p50:** given as the range of the two passes, which differ by up to 28 % (ShareGPT, 6 streams, R793b). The p99 columns of `summary.txt` move by up to 18 % between passes (TPOT p99, ShareGPT, 4 streams) and are not quoted here.
- **Integrity, every cell:**
  - The server's counters over the client's run equal the client's own counts: requests, prompt tokens and generation tokens (84,120 per ShareGPT cell, 122,880 per Spec-Bench cell).
  - 0 failed requests, 0 preemptions, 0 cached prompt tokens on the GPU and 0 from the tiers.
  - The direct clients of the host were stopped for the run and the gateway drained. Every boot of the launcher restarts one proxy, and the driver stopped it again before the cell.

## ShareGPT at 6 streams comes from a re-run of that pair

- **The rule and the miss:** the round's rule, set before it ran and unchanged from the Flash-Next R731, publishes when every level above 1 stream has passes A and B within 3 % on output tok/s. In R793 ShareGPT at 6 streams read 833.9 / 807.9 tok/s, a 3.17 % spread, and the round's decision line is `NOT-PUBLISHABLE sharegpt c6: A/B spread 3.17 % > 3 %`. Every other level was within 1.04 %.
- **Pass B, not pass A, was off:** decode-only steps in pass B took the same time as in pass A (14.26 / 14.28 ms mean). The difference was in the steps that carried other requests' prefill and in time to first token. From 30 to 60 s and 90 to 100 s into pass B's 104 s run, the 120-150 ms prefill-bearing frames that fill every other 10 s window (45 to 107 per window in pass A) fell to 0 to 23, and 40-60 ms and 150-200 ms frames took their place.
- **No cause found:** no other cell of the 28 shows that pattern, which was read from the per-token timings on the serving host (not in the published records). That boot came up on its third attempt after two warmup flakes (`CUDA error: invalid argument`, R233). No GPU clock or power telemetry was recorded during the cells, so the cause is not known.
- **Re-run and decision:** the pair was re-run with the same driver, sample and launcher (R793b, two fresh boots, both on the first attempt). The rule for it was fixed before it was read: take its A/B mean if its spread is at most 3 % and its mean lies within 3 % of R793's pass A. R793b read 829.8 / 835.7 tok/s, spread 0.71 %, mean 832.7, 0.14 % below R793's pass A, so the ShareGPT 6-stream cell is R793b's.
- **What this changes:**
  - R793 applied its run-level rule per cell.
  - This cell was measured in a later session than the other 13. On the Flash-Next stack the same sample differed between sessions by up to 0.9 % per cell (R731 against R731b), more than the within-session spread.
  - R793's own mean of 820.9 would move the plotted point by 1.4 %.

## The disk tier held 3 Spec-Bench prompts and served none of them

The fresh boot empties the GPU cache and the CPU tier but not the disk tier, which stores prompts only, in 1,472-token blocks. Three Spec-Bench prompts reach one block once templated and none reaches two. Pass A's first Spec-Bench cell wrote 57.6 MB to the disk tier; every later cell wrote nothing and read nothing from it (0 external cache hits in all 28 cells). No ShareGPT prompt reaches one block.

## Against the decode curve (R675)

The earlier figure drew these lines dashed over R675's decode curve (2026-09-23, results `2026-09-23-r675-27b-curves`). R675 is `decode_ss.py`: all streams start together, 1,024 forced tokens, one code and one prose prompt, the rate taken over the samples where every stream was decoding.

- **Measurements:**
  - At 16 streams the dashed lines read 1,330 / 1,628 output tok/s and 97 / 111 per stream, against R675's 2,442 / 2,086 decode aggregate and 153 / 130 per stream (ShareGPT / Spec-Bench, code / prose).
  - At 1 stream, with no other request in the batch, the per-stream rates are 224 / 250 against 212 / 169.
  - Spec-Bench stays at or above R675's code line per stream up to 4 streams, and above the prose line up to 8.
- **Conditions that differ between the two lines:**
  - Output length: 210 and 256 against 1,024 tokens, so time to first token and turnover weigh more per output token.
  - Content: MTP τ 2.92 to 3.24 here, against R675's 0.61 to 0.68 of draft tokens accepted on code and 0.46 to 0.49 on prose.
  - Prefill of arriving requests interleaved with decode.
  - Day and boot.
  - Memory clock: offset +4500 in R793. R675 did not record it, and the host's offset read 0 on both cards on 2026-09-25, before the 27B launcher began setting +4500 at every boot.
- **May not be claimed:**
  - The gap between the dashed and solid lines is not a measure of what prefill costs decode. Neither is any ratio of the two lines: at 1 stream, where nothing interleaves, the lines already differ by +6 % to +48 %.
  - Excluding the long frames from TPOT does not isolate decode on these records. On the Flash-Next stack R731 did that; here it overshoots R675's decode aggregate (3,237 against 2,442 at 16 streams), because tokens delivered in the excluded frames stay in the count.
  - vLLM's `max_concurrent_requests` and `max_output_tokens_per_s` fields are not quoted.
