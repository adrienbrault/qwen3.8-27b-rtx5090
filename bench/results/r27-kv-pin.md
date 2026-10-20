# r27-kv-pin: 15.90 GB leaves 1,227 MiB under concurrent image stress (2026-10-20, results `2026-10-20-122821-r27-pin-ladder`, `2026-10-20-131748-r27-headroom-stress`, `2026-10-20-135519-r27-headroom-stress`, `2026-10-20-135353-pin-promote`)

[← all results](../RESULTS.md)

The 2026-10-20 pin ladder and headroom stress use the BA4 image `vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan`, vLLM v0.31.0, no block drop, MTP ns3, NVFP4 KV, bf16 GDN state and 16 sequences, with `PCIE_IPC=1 BSS=1` on the experiment port at stock power. The Quick start's L1a image has no BA4 scratch. Drivers: [r27-pin-ladder.sh](../../scripts/r27-pin-ladder.sh), [r27-headroom-stress.sh](../../scripts/r27-headroom-stress.sh). The headroom rule written before the data requires every request to succeed, no OOM/CUDA-error line and at least 1,000 MiB free on both cards throughout the stress.

## All nine text-stress boots start; 17.90 GB fails headroom

2026-10-20 12:28 to 13:23 UTC, raw [audit and summary](2026-10-20-122821-r27-pin-ladder/). The tier is wiped before each boot; each boot uses fresh seeds and five concurrent `--ctx 120000` cold prefills followed by one `--ctx 250000` cold prefill. Those filler budgets produce about 100K and 208K server-counted prompt tokens. Free VRAM is sampled every 2 s and the large prefill is verified cold. Values below are computed from the audit's per-boot rows.

| pin GB/card | pool tokens | qualifying boots | minimum free per boot (MiB) | verified-cold t/s range |
| --- | --- | --- | --- | --- |
| 15.90 | 1,490,437 | 3/3 | 2631 / 2877 / 2561 | 3,830–3,834 |
| 16.90 | 1,583,674 | 3/3 | 1611 / 1553 / 1181 | 3,831–3,839 |
| 17.90 | 1,676,910 | 0/3 | 415 / 857 / 605 | 3,818–3,834 |

No warmup crash occurred in nine boots. The earlier v0.29 ladder capped the pin at 14.86 GB on boot reliability: 15.90 booted 2/3 and 16.90 1/3 ([R231/R234](r231-promote-nvidia.md), 2026-09-09, results `2026-09-09-r233-nvidia-pin-clean`). The new text-only ladder qualifies 16.90 GB, but it does not exercise vision-encoder memory.

## Concurrent images reject 16.90 GB

2026-10-20 13:24 to 13:42 UTC, raw [audit and summary](2026-10-20-131748-r27-headroom-stress/). The workload starts 16 code streams decoding 1,024 tokens, one `--ctx 120000` cold prefill, one request with 16 screenshots (30,964 prompt tokens) and one with four 4096×4096 noise images (65,564 prompt tokens) concurrently. The driver's old `img12` tag names the 16-image request; the audit's image count is the measured load. Free VRAM is sampled every 0.5 s, 96 samples per arm; BA4 is 96/96 active.

| pin GB/card | min free MiB | c16 decode aggregate t/s | headroom rule |
| --- | --- | --- | --- |
| 16.90 | 99 | 2,588.7 | FAIL |
| 15.90 | 1,227 | 2,683.7 | PASS |

Both arms finish every request with no OOM/CUDA-error lines. The 16.90 GB arm falls to 99 MiB, failing the rule despite its text-only success. The 15.90 GB arm leaves 1,227 MiB and passes. The first daily restore attempt failed with no reason recorded; attempt 2 restored it.

## The 16.02 GB test fails on one of two boots

2026-10-20 13:57 to 14:10 UTC, raw [audit and summary](2026-10-20-135519-r27-headroom-stress/). The same concurrent workload on two boots gives a 1,501,247-token pool and BA4 96/96 active on both.

| pin GB/card | min free MiB | c16 decode aggregate t/s | headroom rule |
| --- | --- | --- | --- |
| 16.02 | 925 | 2,629.0 | FAIL |
| 16.02 | 1,047 | 2,660.6 | PASS |

Every request succeeds and neither boot logs an OOM/CUDA error. One boot leaves 925 MiB, below 1,000, so the two-boot candidate fails; the daily stays at 15.90 GB.

## The promotion boots 1,490,437 pool tokens and passes five tool prompts

2026-10-20 13:53 to 13:57 UTC, raw [promotion log and summary](2026-10-20-135353-pin-promote/). The launcher identity is `5de3072d45817b47255ccfa6ef4d494b`; the public mirror is [serve-v031-served-daily.sh](../../scripts/serve-v031-served-daily.sh). It pins 15,900,000,000 bytes per card at 16 sequences and admits a daily pool from 1,460,000 to 1,525,000 tokens. The promotion reads 1,490,437 tokens, BA4 96/96 active, 2,561 MiB minimum free after pre-warm, 400 W/card after boot and tool smoke 5/5. The 14.86 GB rollback is [serve-v031-nbd-daily.sh](../../scripts/serve-v031-nbd-daily.sh). Pool growth is `(1,490,437 / 1,391,795 − 1) × 100 = 7.1 %`.

The supplied pin export contains audit logs and summaries, plus the promotion log. Individual engine logs, request files and sampled VRAM series are not supplied, so the minima and success counts are verified against the audit summaries rather than independently recomputed from samples. The public drivers retain the serving host's queue and probe dependencies.

## Capacity arithmetic includes the per-request GDN state

The README uses the measured pool and R200's 17,584 tokens-equivalent per running MTP ns3 request (2026-09-05, results `2026-09-05-r200b-pool-cost`, [R200](r200-c32-c64-pool-cost.md)). This is pool accounting derived from a historical state-cost measurement, not a new admission test. Full-window capacity is `floor(1,490,437 / (262,144 + 17,584)) = 5`; at 16 requests it is `1,490,437 / 16 − 17,584 = 75,568.3125` tokens each before block rounding. The earlier pool gives 4 full-window requests and 69,403.1875 tokens each at 16. The new stress does not measure simultaneous full-window admission.
