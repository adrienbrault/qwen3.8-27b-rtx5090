# R211: the prefill chunk ladder

2026-09-08 10:34 to 11:19 UTC, results `2026-09-08-r211-mnbt-ladder`, `scripts/r211-mnbt-ladder.sh`, `scripts/r211-summary.py`.

## `--max-num-batched-tokens` is not the chunk size

The scheduler snaps every non-final prefill chunk down to a multiple of `cache_config.block_size`, because in `align` mamba cache mode the SSM state is materialised at block boundaries. The escape hatch that skips the snap requires a drafter that is not eagle-family, and `SpeculativeConfig.use_eagle()` returns true for method `mtp`, which is the served route. The boot line `SM12X eagle-drop replay boundary retained` confirms it on the running engine.

Block size here is 1,472, so the effective chunk is `floor(MNBT / 1472)` blocks:

| `--max-num-batched-tokens` | effective chunk |
| --- | --- |
| 1536 | 1,472 (1 block) |
| 4096 | 2,944 (2 blocks) |
| 8192 | 7,360 (5 blocks) |
| 16384 | 16,192 (11 blocks) |

Only block multiples are distinct settings. 8192 and 7424 give the same engine.

## What was measured

Six boots on the experiment port, each on the served route (MTP at 3 draft tokens, `pcie_ipc` all-reduce, batch-sharded sampling), each onto a wiped disk tier. Two scored rows per boot, both at an offered concurrency of 20:

- **steady** — 60 requests, a 24,000-token shared prefix plus 8,000 unique tokens, 900-token answers. 75 % prefix reuse, which is the regime the served configuration runs at (it settled at 82.5 % on a real 20-agent coding session the same morning).
- **ramp** — 24 requests, 32,000 unique tokens each, no shared prefix. The cold-burst worst case.

A single-request `plant` row precedes each steady row so the steady row is not measuring its own cold shared prefix.

Seeds are held identical across arms. Every arm is a fresh boot onto a wiped tier, so no arm can inherit another's cache, and a fixed seed removes an output-length work confound: sampled output lengths come from the same generator as the prefix, so per-arm seeds would give each arm a few percent different decode work — the same size as the effect being measured.

The primary score is the delta of the engine's own `inter_token_latency_seconds` histogram across each row, scraped before and after. A stall is an inter-token gap above 0.2 s, which is the same "above five times the median" criterion used earlier, expressed on the histogram grid.

## Result

Arms `A` and `A2` are the same configuration on separate boots and give the noise floor: 1.1 % on wall-clock, 0.07 % on median end-to-end latency, 1.6 % on stall length, 3.6 % relative on stall rate.

**steady row**

| arm | chunk | wall s | stall rate | stall median | mean ITL | TTFT s | median E2EL s | output tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 5 blocks | 128.7 | 10.62 % | 1.095 s | 104 ms | 13.30 | 35.88 | 446.6 |
| B | 2 blocks | 138.0 | 18.48 % | 0.442 s | 97 ms | 13.71 | 36.81 | 412.9 |
| C | 1 block | 135.4 | 37.07 % | 0.250 s | 104 ms | 14.68 | 38.99 | 421.7 |
| D | 11 blocks | 128.7 | 9.26 % | 1.058 s | 102 ms | 13.58 | 35.57 | 446.0 |
| E | 2 blocks, 32 seqs | 133.9 | 23.26 % | 0.443 s | 122 ms | 9.01 | 40.27 | 427.0 |
| A2 | 5 blocks, repeat | 130.1 | 10.25 % | 1.113 s | 102 ms | 13.36 | 35.90 | 441.5 |

**ramp row**

| arm | chunk | wall s | stall rate | stall median | mean ITL | TTFT s | median E2EL s | output tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 5 blocks | 122.4 | 18.20 % | 0.895 s | 181 ms | 40.98 | 100.69 | 170.7 |
| B | 2 blocks | 133.2 | 41.72 % | 0.413 s | 224 ms | 45.07 | 112.92 | 156.2 |
| C | 1 block | 126.3 | 55.88 % | 0.250 s | 162 ms | 49.46 | 82.54 | 165.4 |
| D | 11 blocks | 119.8 | 10.88 % | 1.556 s | 199 ms | 40.95 | 100.25 | 174.9 |
| E | 2 blocks, 32 seqs | 123.6 | 46.28 % | 0.413 s | 206 ms | 44.06 | 96.72 | 168.9 |
| A2 | 5 blocks, repeat | 123.9 | 17.62 % | 0.902 s | 177 ms | 41.41 | 98.18 | 168.6 |

## The stall is one chunk step, and the rate moves the other way

Earlier work established that a decode stall essentially never happens without a prefill in flight. If the stall is one chunk step, stall medians across the 1-, 2-, 5- and 11-block arms must scale 1 : 2 : 5 : 11. On the steady row the measured ratios against the 5-block arm are 0.23, **0.40**, 1.00 and 0.97, against 0.20, **0.40**, 1.00 and 2.20 predicted. The 2-block arm lands exactly.

The 11-block arm breaks the prediction, and the break is informative: the stall is `min(chunk, the request's remaining prefill)`. At 75 % reuse a request has only about 8,000 uncached tokens, so a chunk larger than 5 blocks has nothing left to fill. On the cold ramp, where each request carries 32,000 uncached tokens, the same arm does scale (ratio 1.74).

The cost that a length-only model misses is that halving the chunk roughly **doubles the stall rate** — 10.62 % to 18.48 % on steady, 18.20 % to 41.72 % on ramp. Total prefill work is fixed, so half-size steps mean twice as many decode steps collide with one. Length times rate is what a reader feels, and it goes opposite ways in the two regimes: on steady the 2-block arm spends 36 % less time stalled per token, on ramp it spends 17 % more.

## Choosing a chunk

A 2-block chunk costs 7.5 % of output throughput, 2.6 % of steady end-to-end latency and 12 % of cold-burst end-to-end latency. It buys mean inter-token latency of 97 ms instead of 104 ms, and stalls of 0.44 s instead of 1.10 s. That is a real difference in how steadily output arrives, and it is the wrong trade for work scored on completion rather than on streaming feel. A 1-block chunk is worse than 2 on everything except stall length; its median end-to-end latency on the ramp row is 18 % better than the 5-block arm but its p90 is not, and its median inter-token latency is 205 ms against 26 ms — a fairness shift, not a throughput gain.

Single-request cold prefill of 32,000 tokens is monotone in chunk size — 4.11 s at 11 blocks, 4.23 s at 5, 4.31 s at 2, 4.52 s at 1 — so the kernel-level cost of a small chunk is only about 2 % at 2 blocks. The throughput loss under load is a scheduling effect, not a kernel one.

The served configuration runs 8,192, a 5-block chunk. It was set to 4,096 for 76 minutes on 2026-09-08 and this ladder is why it was set back.

## Time to first token is queue, not chunk

Across every arm, roughly 75 % of time-to-first-token is queue rather than prefill compute. Arm E raises `--max-num-seqs` from 16 to 32 at the same 2-block chunk: steady time-to-first-token falls from 13.71 s to 9.01 s, a 34 % reduction, and queue time falls 36 %, for a 3.8 % smaller KV pool (1,259,372 tokens against 1,309,368). It costs 12 % on median end-to-end latency and 26 % on mean inter-token latency, because twenty admitted streams share the compute that sixteen had. Responses start sooner and finish later.

On the cold ramp the same change does nothing (44.06 s against 45.07 s): admitting twenty requests at once does not create capacity for them.
