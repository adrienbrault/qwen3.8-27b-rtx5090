# r27-served-curves: served decode and prefill (2026-10-20, results `2026-10-20-141816-r27-served-curves`)

[← all results](../RESULTS.md)

2026-10-20 14:18 to 14:37 UTC, raw records in [`2026-10-20-r27-served-curves/`](2026-10-20-r27-served-curves/), driver [`scripts/r27-served-curves.sh`](../../scripts/r27-served-curves.sh), figures drawn by [`bench/plot.py`](../plot.py). The served launcher is [`scripts/serve-v031-served-daily.sh`](../../scripts/serve-v031-served-daily.sh), image `vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan`: vLLM v0.31.0, FlashInfer 0.7.0.post1, NVIDIA NVFP4 weights, NVFP4 KV at 15.90 GB per card, bf16 GDN state, MTP ns3 with `"disable_eagle_block_drop":true`, 16 sequences, a 262,144-token window, BA4, L1a, `pcie_ipc`, batch-sharded sampling, split-KV and the V2 runner. The tiers are 16 GiB host RAM and 300 GB disk with LRU eviction.

The [audit log](2026-10-20-r27-served-curves/audit.log) admits the first boot, records a 1,490,437-token pool and 2,757 / 2,757 MiB free after pre-warm, stock 600 / 575 W limits during measurement, zero engine errors, zero preemptions and zero Xid events. The [boot log](2026-10-20-r27-served-curves/boot-1.log) records BA4 96/96 active and readback of memory offset +4500 and core offset 0 on both cards. The launcher sets a 400 W cap after boot; the driver sets stock limits for these measurements.

The earlier BA4 curves are [r27-ba4-curves](r27-ba4-curves.md), 2026-10-19, results `2026-10-19-233406-r27-ba4-curves`, raw mirror `2026-10-19-r27-ba4-curves/`. That launcher used block drop and the 14.86 GB pin. The comparisons below use `(served / earlier BA4 − 1) × 100` before rounding. These are one boot against one boot, with different acceptance draws; they do not isolate either promotion. The controlled BA4 comparison remains the paired rounds in [r27-ba4-std-bench](r27-ba4-std-bench.md#paired-rounds-against-ba4-switched-off), 2026-10-20, results `2026-10-20-025104-r27-ba4-paired-std`, `2026-10-20-040845-r27-ba4-paired-std`, `2026-10-20-053709-r27-ba4-paired-std`.

## One stream decodes 263.9 t/s of code and 215.9 of prose

`decode_ss.py` synchronizes the streams, forces 1,024 tokens at temperature 0.6, and measures only samples where all streams are decoding. Each row is the median of three runs; aggregate is the measured sum of stream rates. Acceptance is accepted draft tokens divided by drafted tokens from server counters. Sources are `decode-code.jsonl` and `decode-prose.jsonl` in the two dated directories above; the instrument and seeds follow R675 and the earlier BA4 curves.

### Code

| streams | earlier BA4 aggregate | served aggregate | Δ aggregate | served per stream | served acceptance |
| --- | --- | --- | --- | --- | --- |
| 1 | 273.3 | 263.9 | −3.4 % | 263.9 | 0.6300 |
| 2 | 444.2 | 443.1 | −0.2 % | 221.6 | 0.6150 |
| 4 | 937.8 | 949.0 | +1.2 % | 237.2 | 0.6820 |
| 6 | 1,332.4 | 1,251.9 | −6.0 % | 208.7 | 0.6080 |
| 8 | 1,678.1 | 1,669.3 | −0.5 % | 208.7 | 0.6650 |
| 12 | 2,238.2 | 2,287.7 | +2.2 % | 190.6 | 0.6720 |
| 16 | 2,646.7 | 2,648.6 | +0.1 % | 165.5 | 0.6230 |

### Prose

| streams | earlier BA4 aggregate | served aggregate | Δ aggregate | served per stream | served acceptance |
| --- | --- | --- | --- | --- | --- |
| 1 | 225.3 | 215.9 | −4.2 % | 215.9 | 0.4600 |
| 2 | 379.9 | 377.0 | −0.8 % | 188.5 | 0.4680 |
| 4 | 772.8 | 766.9 | −0.8 % | 191.7 | 0.4880 |
| 6 | 1,047.6 | 1,071.8 | +2.3 % | 178.6 | 0.4730 |
| 8 | 1,355.1 | 1,350.0 | −0.4 % | 168.8 | 0.4750 |
| 12 | 1,849.4 | 1,860.0 | +0.6 % | 155.0 | 0.4820 |
| 16 | 2,232.2 | 2,239.7 | +0.3 % | 140.0 | 0.4760 |

Code c1 spans 256.0 to 275.8 t/s and prose c1 201.3 to 219.2. Code c6 reads −6.0 %, code c12 +2.2 % and prose c6 +2.3 % against the earlier boot; the other c4–c16 rows are within 1.2 %. The BA4 M=4 kernel serves one stream with three draft tokens; calls with more rows run the original GEMM.

## Cold prefill reaches 3,994 t/s at 200K prompt tokens

`kv_capacity_probe.py --conc 1 --tokens 1` sends three fresh-seeded prompts per length, one at a time. `prefill.jsonl` records server-counted prompt tokens and one-token latency to 0.1 s. TTFT is their mean latency; rate is the mean of the three individual token/latency rates. No preemptions are recorded.

| filler budget `--ctx` | mean prompt tokens | TTFT (s) | prefill (t/s) |
| --- | --- | --- | --- |
| 8,000 | 6,690 | 0.70 | 9,557 |
| 30,000 | 25,024 | 2.93 | 8,533 |
| 60,000 | 50,067 | 6.77 | 7,399 |
| 120,000 | 99,966 | 17.27 | 5,790 |
| 200,000 | 166,549 | 37.17 | 4,481 |
| 240,000 | 200,078 | 50.10 | 3,994 |

## One code stream reads 224.1 t/s at a 200K filler budget

`decode_ss.py --conc 1 --ctx N` prepends N/1.3 filler words; prompt tokens are not recorded, so the figure uses filler budget. Each point is the median of two runs from `decode-{code,prose}-c1-{30,60,120,200}k.jsonl`; the zero-filler point reuses c1 above. These runs do not follow the cold-prefill protocol.

| filler budget | code t/s | prose t/s | code acceptance | prose acceptance |
| --- | --- | --- | --- | --- |
| 30,000 | 293.6 | 217.6 | 0.7880 | 0.5005 |
| 60,000 | 271.5 | 212.7 | 0.7540 | 0.5170 |
| 120,000 | 243.8 | 192.6 | 0.7295 | 0.5065 |
| 200,000 | 224.1 | 177.8 | 0.7515 | 0.5275 |

## GPU draw exceeds the served cap during deep decode

`power-summary.json` records NVML GPU energy at stock 600 / 575 W limits. The driver orders 42 concurrency phases followed by 16 depth phases; grouping that order gives three runs per concurrency row and two per depth row. Each column is its own median. CPU RAPL and wall power are excluded; wall power was not metered.

| shape | code GPU W (0 / 1) | prose GPU W (0 / 1) | code J/token | prose J/token |
| --- | --- | --- | --- | --- |
| 1 stream(s) | 660 (345 / 316) | 678 (356 / 322) | 2.502 | 3.140 |
| 2 stream(s) | 599 (314 / 284) | 623 (329 / 294) | 1.351 | 1.652 |
| 4 stream(s) | 640 (334 / 306) | 648 (341 / 308) | 0.675 | 0.845 |
| 6 stream(s) | 655 (342 / 311) | 663 (348 / 316) | 0.519 | 0.620 |
| 8 stream(s) | 654 (345 / 309) | 663 (346 / 316) | 0.389 | 0.491 |
| 12 stream(s) | 725 (381 / 344) | 728 (383 / 347) | 0.316 | 0.391 |
| 16 stream(s) | 721 (378 / 340) | 726 (382 / 345) | 0.270 | 0.324 |
| 1 stream(s), 30K filler | 747 (393 / 353) | 729 (383 / 346) | 2.546 | 3.352 |
| 1 stream(s), 60K filler | 774 (412 / 362) | 763 (404 / 358) | 2.852 | 3.587 |
| 1 stream(s), 120K filler | 826 (438 / 387) | 822 (435 / 387) | 3.390 | 4.270 |
| 1 stream(s), 200K filler | 876 (463 / 413) | 861 (455 / 406) | 3.912 | 4.840 |

Card 0 exceeds 400 W at 60K filler and deeper; the depth rates are stock-limit rates. The separate 400 W comparison predates the pin promotion ([r27-power-cap](r27-power-cap.md), 2026-10-20, results `2026-10-20-082818-r27-power-curves`).
