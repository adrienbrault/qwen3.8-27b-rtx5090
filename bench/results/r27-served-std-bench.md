# r27-served-std-bench: served cells with disk-tier hits (2026-10-20, results `2026-10-20-r27-served-std-bench-1418`)

[← all results](../RESULTS.md)

2026-10-20 14:38 to 17:20 UTC, raw records in [`2026-10-20-r27-served-std-bench-1418/`](2026-10-20-r27-served-std-bench-1418/), driver [`scripts/r27-served-std-bench.sh`](../../scripts/r27-served-std-bench.sh). This is the served launcher [`scripts/serve-v031-served-daily.sh`](../../scripts/serve-v031-served-daily.sh), private launcher identity `5de3072d45817b47255ccfa6ef4d494b`, image `vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan`, with no block drop and the 15.90 GB/card pin: 1,490,437 pool tokens, 16 sequences, 262,144-token window, MTP ns3, bf16 GDN state, NVFP4 KV, BA4, L1a, `pcie_ipc`, batch-sharded sampling and 16 GiB CPU / 300 GB disk tiers.

## Thirteen Spec-Bench cells exceed the 1 % cached-token gate

The [summary](2026-10-20-r27-served-std-bench-1418/summary.txt) gives this verdict verbatim:

> DECISION: NOT-PUBLISHABLE 13 cells: cached # % of prompt tokens > # % [A-specbench-c2, A-specbench-c4, A-specbench-c6, A-specbench-c8, A-specbench-c12, A-specbench-c16, B-specbench-c1, B-specbench-c2, B-specbench-c4, B-specbench-c6, B-specbench-c8, B-specbench-c12, B-specbench-c16]; e.g. A-specbench-c2: cached 2.19 % of prompt tokens (- requests) > 1.00 %

The [runs manifest](2026-10-20-r27-served-std-bench-1418/runs.tsv) has all 28 cells with return code 0. The [audit log](2026-10-20-r27-served-std-bench-1418/audit.log) records 0 GPU + 2,944 tier-cached tokens out of 134,377 prompt tokens, 2.19 %, in pass A c2–c16 and every pass B Spec-Bench cell. Pass A c1 has zero cached tokens. All 14 ShareGPT cells have zero cached tokens. All cells completed their 400 or 480 requests with zero failures, preemptions, engine errors, tracebacks or OOM lines; server and client request/token counts agree and [foreign.tsv](2026-10-20-r27-served-std-bench-1418/foreign.tsv) records no foreign requests. The maximum A/B output spread is 1.86 % at ShareGPT c6; spread does not override the cache gate.

Two Spec-Bench prompts exceed one 1,472-token tier block: the client input lengths are 1,491 and 1,498 tokens, and the [sample manifest](2026-10-20-r27-served-std-bench-1418/A-specbench-c1.samples.tsv) independently has two prompts over a block. The audit's templated preflight counts three prompts at least one block long. The first cell populates the persistent disk tier; subsequent cells serve two full blocks, 2 × 1,472 = 2,944 tokens. No block drop retains those hits. With block drop, the last matching block is dropped and recomputed, explaining the zero-hit counters of the earlier BA4 round ([r27-ba4-std-bench](r27-ba4-std-bench.md), 2026-10-19/20, results `2026-10-19-r27-ba4-std-bench-2335`). This mechanism follows the launcher's prefix-hit behavior and the recorded counts; the supplied records contain no per-prompt tier trace. The hits are a property of the served configuration, and these cells are published with their cache condition and the gate's rejection.

All 28 boots were admitted on the first try with the same image, launcher hash, 48-entry environment identity, pool and sequence limit ([boots.tsv](2026-10-20-r27-served-std-bench-1418/boots.tsv)). Free VRAM after boot spans 2,003 to 2,883 MiB/card. The supplied standard-benchmark export does not record per-boot BA4 ACTIVE counts; the image, environment identity and launcher match the served configuration, whose curves and promotion logs record 96/96. Measurement power is stock 600 / 575 W; memory offset +4500 and core offset 0 were read back at every boot and after each cell. The launcher applies 400 W after boot and the driver then restores stock power for measurement.

## The two-pass tables use the same standard benchmark protocol

The client is `vllm bench serve` v0.30.0 through the [Flash-Next request shim](https://github.com/adrienbrault/qwen3.8-flash-next-2x-rtx5090/blob/main/bench/vllm_bench_tabby.py): greedy, streamed, thinking on, closed loop at 1, 2, 4, 6, 8, 12 and 16 requests. ShareGPT V3 uses 400 conversations, seed 7310, forcing each reference reply's length with `min_tokens` (84,120 output tokens per cell, mean 210.3). Spec-Bench uses all 480 questions and 256 forced tokens per request (122,880 output tokens). Each cell has a fresh engine boot, out-of-set warm-ups, then a counter snapshot; the disk tier persists. Pass A runs all cells before pass B. The outputs are truncated reasoning, not scored answers.

Each table value is the mean of the two pass values. Output tok/s includes prefill, TTFT and request turnover over wall time. Per-stream tok/s is 1000 / TPOT p50, calculated per pass before averaging, never multiplied by concurrency to form an aggregate. TPOT is `(latency − TTFT) / (output_tokens − 1)` for each request. Latency quantiles are averaged per-pass quantiles, not pooled quantiles. τ uses the server's `generation_tokens / (generation_tokens − accepted_draft_tokens)`. The public JSONs omit only `generated_texts` and `itls`, following the earlier public rounds; the latency, length, sample manifest and accounting records remain. The supplied export lacks the `server/` snapshots named by runs.tsv and the individual boot/container logs, so accounting and BA4 counts are verified against audit.log rather than those files.

### ShareGPT V3

| streams | wall-clock output tok/s | per stream | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | τ | A/B output spread | cached prompt tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 249.7 | 273.2 | 40.7 / 111.7 | 3.66 / 5.35 | 2.922 | 0.12 % | 0 % |
| 2 | 406.3 | 226.3 | 90.8 / 143.6 | 4.42 / 7.03 | 2.936 | 0.12 % | 0 % |
| 4 | 732.2 | 208.2 | 109.3 / 198.8 | 4.80 / 10.73 | 2.918 | 0.58 % | 0 % |
| 6 | 970.1 | 183.4 | 111.6 / 261.0 | 5.45 / 9.18 | 2.930 | 1.86 % | 0 % |
| 8 | 1,163.1 | 164.2 | 113.5 / 302.9 | 6.09 / 14.04 | 2.943 | 0.30 % | 0 % |
| 12 | 1,440.8 | 135.7 | 117.6 / 358.8 | 7.37 / 16.77 | 2.929 | 0.48 % | 0 % |
| 16 | 1,649.9 | 115.7 | 124.0 / 500.3 | 8.64 / 21.61 | 2.950 | 1.02 % | 0 % |

### Spec-Bench (2.19 % tier-cached in 13 cells; A c1 0 %, B c1 2.19 %)

| streams | wall-clock output tok/s | per stream | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | τ | A/B output spread | cached prompt tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 278.9 | 302.2 | 35.3 / 154.5 | 3.31 / 4.35 | 3.228 | 0.08 % | A 0 % / B 2.19 % |
| 2 | 456.7 | 250.0 | 55.7 / 174.2 | 4.00 / 5.49 | 3.235 | 0.16 % | 2.19 % |
| 4 | 842.3 | 231.7 | 57.9 / 187.2 | 4.32 / 5.90 | 3.242 | 0.70 % | 2.19 % |
| 6 | 1,126.0 | 205.1 | 64.3 / 202.4 | 4.88 / 6.68 | 3.240 | 0.40 % | 2.19 % |
| 8 | 1,359.4 | 184.9 | 76.9 / 215.3 | 5.41 / 7.47 | 3.231 | 0.35 % | 2.19 % |
| 12 | 1,712.4 | 154.9 | 116.0 / 243.4 | 6.46 / 9.25 | 3.232 | 0.24 % | 2.19 % |
| 16 | 1,925.1 | 130.3 | 127.2 / 285.6 | 7.67 / 10.67 | 3.228 | 0.06 % | 2.19 % |

## The figure pairs two instruments on the served configuration

The solid lines in `docs/img/decode-scaling-16.svg` are [r27-served-curves](r27-served-curves.md), 2026-10-20, results `2026-10-20-141816-r27-served-curves`: synchronized streams, one code or prose prompt, 1,024 forced output tokens, decode-only intervals, one boot. The dashed lines use this round, fresh boots and interleaved prefill; the Spec-Bench line carries the 2.19 % condition above. Output lengths, prompts, turnover and boots differ, so the gap does not isolate the cost of prefill. These new absolute readings do not replace the controlled BA4 comparison in the earlier paired rounds.
