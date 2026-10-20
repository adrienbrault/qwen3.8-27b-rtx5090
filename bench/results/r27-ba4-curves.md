# r27-ba4-curves: BA4 decode and prefill curves (2026-10-19, results `2026-10-19-233406-r27-ba4-curves`)

[← all results](../RESULTS.md)

2026-10-19 23:37 to 23:53 UTC, raw records in [`2026-10-19-r27-ba4-curves/`](2026-10-19-r27-ba4-curves/), driver [`scripts/r27-ba4-curves.sh`](../../scripts/r27-ba4-curves.sh), figures drawn by [`bench/plot.py`](../plot.py). The served launcher is [`scripts/serve-v031-ba4-daily.sh`](../../scripts/serve-v031-ba4-daily.sh), image `vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan`, with `VLLM_SM12X_BA_M4=1` and `VLLM_SM12X_DRAFT_UB_PLAN=1`: vLLM v0.31.0, FlashInfer 0.7.0.post1, nvidia/Qwen3.8-27B-NVFP4, NVFP4 KV pinned at 14.86 GB per card, bf16 linear-attention state, MTP at 3 draft tokens, 16 sequences and a 262,144-token window, `pcie_ipc` all-reduce, batch-sharded sampling, split-KV override and the V2 model runner. The KV tiers are 16 GiB of host RAM and a 300 GB LRU-capped disk tier, kept across restarts.

The [audit log](2026-10-19-r27-ba4-curves/audit.log) records first-attempt boot admission, a 1,391,795-token pool, 3,919 and 3,921 MiB free on the two cards after the launcher's pre-warm, stock power limits of 600 / 575 W while measuring, zero engine errors, zero preemptions and zero GPU Xid events. The [boot log](2026-10-19-r27-ba4-curves/boot-1.log) records memory clock offset +4500 and core offset 0, read back on both cards, and `BA4 96/96 active`.

The L1a arm is [`r27-l1a-curves`](r27-l1a-curves.md), measured 2026-10-17 07:12 to 07:30 UTC, results `2026-10-17-r27-l1a-curves`, with raw records in [`2026-10-17-r27-l1a-curves/`](2026-10-17-r27-l1a-curves/), launcher [`scripts/serve-v031-l1a-daily.sh`](../../scripts/serve-v031-l1a-daily.sh), image `vllm-qwen38:v0310-r27c-A_S-L1a-ubplan`. It used the same checkpoint, pin, sequence limit, dtypes, MTP head and tiers at stock power; its memory clock offset was not recorded. Every table below compares those two dated directories; throughput and TTFT deltas are `(BA4 / L1a − 1) × 100`, acceptance deltas are percentage points. Deltas use the raw summary values before display rounding.

These curves use the same instruments, shapes, run counts and seed scheme as [r27-l1a-curves](r27-l1a-curves.md) and [R675's method](r675-27b-curves.md). They describe one L1a boot and one BA4 boot on different days; they cannot isolate the layer's gain from boot variation, prompts, acceptance or clocks. The paired order-balanced screens in [r27 BA4](r27-ba4.md), measured 2026-10-19 in results `2026-10-19-r27-audition-screen-070155-3532152` and `2026-10-19-r27-audition-screen-075049-955439`, are the controlled comparison.

The BA4 kernel serves the `in_proj_ba` call only when the verify step carries four rows, which is one sequence with its three draft tokens. At two or more concurrent streams every call has eight or more rows and runs the original GEMM, as on L1a.

## One stream decodes 273.3 t/s of code and 225.3 of prose

[`decode_ss.py`](../../scripts/decode_ss.py) forces 1,024 tokens per stream at temperature 0, starts all streams together and measures the samples where every stream was decoding, three runs per shape. Each value is the median of the runs; aggregate is the measured sum of stream rates, and per-stream rate is reported separately, in tokens per second. Sources: `decode-code.jsonl` and `decode-prose.jsonl` in each directory above.

### Code

| streams | L1a aggregate | BA4 aggregate | Δ aggregate | L1a per stream | BA4 per stream | Δ per stream |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 241.7 | 273.3 | +13.1 % | 241.7 | 273.3 | +13.1 % |
| 2 | 457.8 | 444.2 | −3.0 % | 228.9 | 222.1 | −3.0 % |
| 4 | 924.7 | 937.8 | +1.4 % | 231.2 | 234.4 | +1.4 % |
| 6 | 1,346.8 | 1,332.4 | −1.1 % | 224.5 | 222.1 | −1.1 % |
| 8 | 1,647.0 | 1,678.1 | +1.9 % | 205.9 | 209.8 | +1.9 % |
| 12 | 2,259.4 | 2,238.2 | −0.9 % | 188.3 | 186.5 | −1.0 % |
| 16 | 2,677.5 | 2,646.7 | −1.2 % | 167.3 | 165.4 | −1.1 % |

### Prose

| streams | L1a aggregate | BA4 aggregate | Δ aggregate | L1a per stream | BA4 per stream | Δ per stream |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 190.4 | 225.3 | +18.3 % | 190.4 | 225.3 | +18.3 % |
| 2 | 391.5 | 379.9 | −3.0 % | 195.8 | 189.9 | −3.0 % |
| 4 | 759.3 | 772.8 | +1.8 % | 189.8 | 193.2 | +1.8 % |
| 6 | 1,049.7 | 1,047.6 | −0.2 % | 175.0 | 174.6 | −0.2 % |
| 8 | 1,348.9 | 1,355.1 | +0.5 % | 168.6 | 169.4 | +0.5 % |
| 12 | 1,858.8 | 1,849.4 | −0.5 % | 154.9 | 154.1 | −0.5 % |
| 16 | 2,228.5 | 2,232.2 | +0.2 % | 139.3 | 139.5 | +0.1 % |

The one-stream rows are the only cells where the kernel runs. The code c1 runs on the BA4 boot spread from 234.0 to 284.9 t/s (L1a: 239.4 to 252.3). Prose c1 acceptance read 0.504 on BA4 against 0.463 on L1a, so part of its +18.3 % is acceptance; the screens measured +11.2 and +13.8 % on prose. At two streams both kinds read −3.0 %, on one boot per arm, with the BA4 kernel not in use at that row count; from four to sixteen streams the deltas lie between −1.2 and +1.9 %.

## Cold prefill matches L1a

[`kv_capacity_probe.py --conc 1 --tokens 1`](../../scripts/kv_capacity_probe.py) submits one request at a time, with three fresh-seeded prompts per length so neither the GPU prefix cache nor the CPU and disk tiers hold them. Sources: `prefill.jsonl` in each directory above. Prompt tokens are server counts; `--ctx` is the requested filler budget. TTFT is the one-token request latency, averaged over the three prompts, whose individual latencies are recorded to 0.1 s. Prefill rate is the mean of the three individual `prompt_tokens / latency` rates, as plotted.

| `--ctx` | L1a mean prompt tokens | BA4 mean prompt tokens | Δ tokens | L1a TTFT (s) | BA4 TTFT (s) | Δ TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,000 | 6,697 | 6,708 | +11 | 0.70 | 0.70 | +0.0 % |
| 30,000 | 25,022 | 25,051 | +29 | 2.93 | 3.00 | +2.3 % |
| 60,000 | 49,950 | 50,037 | +87 | 6.80 | 6.80 | +0.0 % |
| 120,000 | 99,997 | 100,062 | +64 | 17.30 | 17.30 | +0.0 % |
| 200,000 | 166,658 | 166,683 | +25 | 37.33 | 37.40 | +0.2 % |
| 240,000 | 200,049 | 200,080 | +31 | 50.10 | 50.07 | −0.1 % |

| `--ctx` | L1a prefill (t/s) | BA4 prefill (t/s) | Δ rate |
| ---: | ---: | ---: | ---: |
| 8,000 | 9,567 | 9,582 | +0.2 % |
| 30,000 | 8,532 | 8,350 | −2.1 % |
| 60,000 | 7,346 | 7,358 | +0.2 % |
| 120,000 | 5,780 | 5,784 | +0.1 % |
| 200,000 | 4,464 | 4,457 | −0.2 % |
| 240,000 | 3,993 | 3,996 | +0.1 % |

The 30K row is three latencies of 3.0 s on BA4 against 2.9, 3.0 and 2.9 s on L1a, one step of the 0.1 s recording resolution. Prefill calls carry thousands of rows per GEMM, where the layer runs the original GEMM.

## One code stream reads 219.6 t/s at a 200K filler budget

`decode_ss.py --conc 1 --ctx N` puts N/1.3 filler words before the prompt, two runs per point. The probe does not record the prompt's token count, so the depth figure uses the filler budget rather than prompt tokens. Sources: `decode-{code,prose}-c1-{30,60,120,200}k.jsonl` in each directory above; the zero-filler point is the three-run c1 row from the concurrency sweep. Values below are per-stream decode tokens per second.

| `--ctx` | L1a code | BA4 code | Δ code | L1a prose | BA4 prose | Δ prose |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 241.7 | 273.3 | +13.1 % | 190.4 | 225.3 | +18.3 % |
| 30,000 | 246.5 | 276.0 | +12.0 % | 184.6 | 215.8 | +16.9 % |
| 60,000 | 235.7 | 259.4 | +10.1 % | 184.8 | 211.7 | +14.6 % |
| 120,000 | 216.2 | 238.2 | +10.2 % | 175.2 | 194.2 | +10.8 % |
| 200,000 | 209.1 | 219.6 | +5.0 % | 158.8 | 181.2 | +14.1 % |

On the BA4 boot, code at 200K reads 19.6 % below its zero-filler rate and prose 19.6 % below. At 200K the code runs accepted 0.731 of the draft tokens on BA4 against 0.785 on L1a. The depth runs do not use the cold-prefill protocol: their TTFT values are not cold-prefill measurements.

## Draft acceptance differs by text and depth

Acceptance is accepted draft tokens divided by drafted tokens from the server counters, reported as a fraction and taken as the median over runs. The concurrency rows use three runs and the depth rows two; the zero-filler depth row reuses c1. Sources are the same decode JSONL summaries as the throughput tables.

### Acceptance against concurrency

| streams | L1a code | BA4 code | Δ code | L1a prose | BA4 prose | Δ prose |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6790 | 0.6650 | −1.40 pp | 0.4630 | 0.5040 | +4.10 pp |
| 2 | 0.6430 | 0.6220 | −2.10 pp | 0.5030 | 0.4780 | −2.50 pp |
| 4 | 0.6560 | 0.6700 | +1.40 pp | 0.4850 | 0.5010 | +1.60 pp |
| 6 | 0.6790 | 0.6840 | +0.50 pp | 0.4600 | 0.4650 | +0.50 pp |
| 8 | 0.6520 | 0.6780 | +2.60 pp | 0.4740 | 0.4830 | +0.90 pp |
| 12 | 0.6590 | 0.6520 | −0.70 pp | 0.4810 | 0.4770 | −0.40 pp |
| 16 | 0.6340 | 0.6260 | −0.80 pp | 0.4740 | 0.4710 | −0.30 pp |

### Acceptance at depth

| `--ctx` | L1a code | BA4 code | Δ code | L1a prose | BA4 prose | Δ prose |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.6790 | 0.6650 | −1.40 pp | 0.4630 | 0.5040 | +4.10 pp |
| 30,000 | 0.7365 | 0.7260 | −1.05 pp | 0.4650 | 0.4990 | +3.40 pp |
| 60,000 | 0.7305 | 0.7125 | −1.80 pp | 0.5005 | 0.5195 | +1.90 pp |
| 120,000 | 0.7215 | 0.7075 | −1.40 pp | 0.5200 | 0.5170 | −0.30 pp |
| 200,000 | 0.7850 | 0.7305 | −5.45 pp | 0.5165 | 0.5445 | +2.80 pp |

## GPU power during decode

[`power-summary.json`](2026-10-19-r27-ba4-curves/power-summary.json) holds the unit's NVML energy per measured phase, both cards, at stock limits of 600 / 575 W; the served launcher caps both cards at 400 W after boot. The CPU package (RAPL) drew about 100 to 115 W in every phase and is not included below; wall power was not metered. Each row is the median over the phase's runs.

| shape | code GPU W (card 0 / card 1) | code J/token | prose GPU W (card 0 / card 1) | prose J/token |
| --- | ---: | ---: | ---: | ---: |
| 1 stream | 648 (343 / 306) | 2.416 | 669 (351 / 319) | 2.961 |
| 2 streams | 613 (322 / 289) | 1.379 | 621 (329 / 292) | 1.646 |
| 4 streams | 638 (337 / 301) | 0.680 | 652 (343 / 308) | 0.844 |
| 6 streams | 644 (338 / 307) | 0.486 | 649 (340 / 309) | 0.619 |
| 8 streams | 656 (346 / 310) | 0.393 | 658 (345 / 312) | 0.486 |
| 12 streams | 719 (378 / 342) | 0.318 | 721 (380 / 341) | 0.390 |
| 16 streams | 721 (379 / 341) | 0.272 | 724 (381 / 344) | 0.324 |
| 1 stream, 30K filler | 738 (388 / 350) | 2.677 | 724 (380 / 344) | 3.356 |
| 1 stream, 60K filler | 771 (410 / 362) | 2.973 | 762 (406 / 356) | 3.598 |
| 1 stream, 120K filler | 830 (440 / 390) | 3.485 | 821 (436 / 385) | 4.229 |
| 1 stream, 200K filler | 870 (459 / 411) | 3.961 | 883 (470 / 413) | 4.881 |

At 60K filler and deeper, card 0 draws above the served 400 W cap, so the depth rates above are stock-limit rates; at no filler and at 1 to 16 streams both cards stay below it.
