# R159, the served shape at concurrency 8/16/32/64: aggregate saturates at c16 and the pool caps admission at ~18 short requests (2026-09-02, `results/2026-09-02-r159-conc-b`, `scripts/r159-conc.sh` + `r159c-live.sh`)

[← all results](../RESULTS.md)

The daily (the served configuration) admits 8 streams (`--max-num-seqs 8`). The same shape (RedHat NVFP4 weights, fp8 KV, DFlash2 ns9, TP=2, MNBT 8192, no tier) was booted with `--max-num-seqs 64`. At util 0.92 it OOM'd on the first c64 step: `sample_tokens` needs 64 × 10 spec positions × vocab × fp32 = 392 MB of logits the boot profiler never budgets. Util 0.90 (pool 624,284) survived with three allocator OOM-retry warnings during c64.

llama-benchy pp2048/tg256, T=0.6, 3 runs, one boot:

| conc | prefill t/s | decode peak agg t/s | TTFT |
|---|---|---|---|
| 8 | 9,235 | 1,467 | 1.32 s |
| 16 | 9,174 | **2,012** | 2.20 s |
| 32 | 6,604 | 1,934 | 5.01 s |
| 64 | 5,989 | 1,990 | 10.7 s |

Steady-state decode (`decode_ss.py`, window = samples with `num_requests_running == c`, 512 tokens, 3 runs): code aggregate measured 1,147 at c8 (143/stream, accept 0.27) and 1,533 at c16 (96/stream, 0.30). Prose measured 914 at c8 (114, 0.20) and 1,213 at c16 (76, 0.21). c32 and c64 have no steady state even with 2,048-token outputs: `num_requests_running` never exceeded **17**, the rest waited on `capacity`, `kv_cache_usage_perc` read 0.88 with 15 running ~2K-token requests, 38 preemptions.

Why 17, and what the run does and does not establish. Measured: the cap itself, and that each request's pool cost is large and mostly fixed (the pool admits 2.38 streams at 262K and ~17 at ~2K). Layout, from the boot log: vLLM sets the attention block to 1,664 tokens so one attention page equals one mamba page, and the two padding warnings (16 attention → 20, 48 GDN → 50) mean the group size is 5, the DFlash drafter's layer count, whose KV shares the pool. That gives 15 KV-cache groups, each request holding at least one block in every group, and the mamba page also carries the speculative-decode state slots. How that adds up to exactly ~17 is not isolated: the usage gauge includes evictable prefix-cache blocks, so it cannot be read as a per-request floor. A fresh boot with prefix caching off and a c=8..24 ramp would settle it. What stands regardless: on this shape admission is set by the pool, not by `--max-num-seqs`.

For the daily: raising `--max-num-seqs` to 16 sits under the measured short-request ceiling and adds ~+34% aggregate with no queueing for streams 9–16. The cost is per-stream 143 → 96 t/s on code during bursts, and at deep context preemption storms instead of queueing. Not promoted.
