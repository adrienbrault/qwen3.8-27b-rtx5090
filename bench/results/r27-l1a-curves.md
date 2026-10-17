# r27-l1a-curves: L1a decode and prefill curves (2026-10-17, results `2026-10-17-r27-l1a-curves`)

[← all results](../RESULTS.md)

2026-10-17 07:12 to 07:30 UTC, raw records in [`2026-10-17-r27-l1a-curves/`](2026-10-17-r27-l1a-curves/), driver [`scripts/r27-l1a-curves.sh`](../../scripts/r27-l1a-curves.sh), figures drawn by [`bench/plot.py`](../plot.py). The served launcher is [`scripts/serve-v031-l1a-daily.sh`](../../scripts/serve-v031-l1a-daily.sh), image `vllm-qwen38:v0310-r27c-A_S-L1a-ubplan`, with `VLLM_SM12X_DRAFT_UB_PLAN=1`: vLLM v0.31.0, FlashInfer 0.7.0.post1, nvidia/Qwen3.8-27B-NVFP4, NVFP4 KV pinned at 14.86 GB per card, bf16 linear-attention state, MTP at 3 draft tokens, 16 sequences and a 262,144-token window, `pcie_ipc` all-reduce, batch-sharded sampling, split-KV override and the V2 model runner. The KV tiers are 16 GiB of host RAM and a 300 GB LRU-capped disk tier, kept across restarts.

The [audit log](2026-10-17-r27-l1a-curves/audit.log) records first-attempt boot admission, a 1,391,795-token pool, 3,613 MiB free on each card after the launcher's pre-warm, stock power limits of 600 / 575 W, zero engine errors, zero preemptions and zero GPU Xid events. The memory clock offset was not recorded.

The A_S arm is [`r27-as-curves`](r27-as-curves.md), measured 2026-10-16 20:32 to 20:49 UTC, results `2026-10-16-201838-r27-as-curves`, with raw records in [`2026-10-16-r27-as-curves/`](2026-10-16-r27-as-curves/), launcher [`scripts/serve-v031-splitkv-daily.sh`](../../scripts/serve-v031-splitkv-daily.sh), image `vllm-qwen38:v0310-r27c-A_S-splitkv`. It used the same checkpoint, pin, sequence limit, dtypes, MTP head and tiers, also at stock power with no recorded memory clock offset. Every table below compares those two dated directories; throughput and TTFT deltas are `(L1a / A_S − 1) × 100`, acceptance deltas are percentage points. Deltas use the raw summary values before display rounding.

These curves use the same instruments, shapes, run counts and seed scheme as [R675's method](r675-27b-curves.md) and [r27-as-curves](r27-as-curves.md), with different seed prefixes. They describe the rates on one A_S boot and one L1a boot on different days; they cannot isolate the layer's causal gain from boot variation, prompts, acceptance or unrecorded clocks. The paired order-balanced [r27 L1a screen](r27-l1a.md), measured 2026-10-17 in results `2026-10-17-r27-audition-screen-022747-4123606`, is the controlled comparison.

## Decode aggregate rises through 16 streams

[`decode_ss.py`](../../scripts/decode_ss.py) forces 1,024 tokens per stream at temperature 0, starts all streams together and measures the samples where every stream was decoding, three runs per shape. Each value is the median of the runs; aggregate is the measured sum of stream rates, and per-stream rate is reported separately, in tokens per second. Sources: `decode-code.jsonl` and `decode-prose.jsonl` in each directory above.

### Code

| streams | A_S aggregate | L1a aggregate | Δ aggregate | A_S per stream | L1a per stream | Δ per stream |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 229.5 | 241.7 | +5.3 % | 229.5 | 241.7 | +5.3 % |
| 2 | 446.8 | 457.8 | +2.5 % | 223.4 | 228.9 | +2.5 % |
| 4 | 888.3 | 924.7 | +4.1 % | 222.1 | 231.2 | +4.1 % |
| 6 | 1,214.8 | 1,346.8 | +10.9 % | 202.5 | 224.5 | +10.9 % |
| 8 | 1,609.7 | 1,647.0 | +2.3 % | 201.2 | 205.9 | +2.3 % |
| 12 | 2,172.6 | 2,259.4 | +4.0 % | 181.0 | 188.3 | +4.0 % |
| 16 | 2,633.4 | 2,677.5 | +1.7 % | 164.6 | 167.3 | +1.6 % |

### Prose

| streams | A_S aggregate | L1a aggregate | Δ aggregate | A_S per stream | L1a per stream | Δ per stream |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 189.6 | 190.4 | +0.4 % | 189.6 | 190.4 | +0.4 % |
| 2 | 360.6 | 391.5 | +8.6 % | 180.3 | 195.8 | +8.6 % |
| 4 | 712.4 | 759.3 | +6.6 % | 178.1 | 189.8 | +6.6 % |
| 6 | 1,006.0 | 1,049.7 | +4.3 % | 167.7 | 175.0 | +4.4 % |
| 8 | 1,362.2 | 1,348.9 | −1.0 % | 170.3 | 168.6 | −1.0 % |
| 12 | 1,813.6 | 1,858.8 | +2.5 % | 151.1 | 154.9 | +2.5 % |
| 16 | 2,205.8 | 2,228.5 | +1.0 % | 137.9 | 139.3 | +1.0 % |

## Cold prefill spans 6.7K to 200K measured prompt tokens

[`kv_capacity_probe.py --conc 1 --tokens 1`](../../scripts/kv_capacity_probe.py) submits one request at a time, with three fresh-seeded prompts per length so neither the GPU prefix cache nor the CPU and disk tiers hold them. Sources: `prefill.jsonl` in each directory above. Prompt tokens are server counts; `--ctx` is the requested filler budget. TTFT is the one-token request latency, averaged over the three prompts, whose individual latencies are recorded to 0.1 s; the displayed mean does not imply finer individual timing. Prefill rate is the mean of the three individual `prompt_tokens / latency` rates, as plotted, rather than mean tokens divided by mean latency.

| `--ctx` | A_S mean prompt tokens | L1a mean prompt tokens | Δ tokens | A_S TTFT (s) | L1a TTFT (s) | Δ TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,000 | 6,680 | 6,697 | +17 | 0.70 | 0.70 | +0.0 % |
| 30,000 | 24,984 | 25,022 | +38 | 2.90 | 2.93 | +1.1 % |
| 60,000 | 50,041 | 49,950 | −91 | 6.80 | 6.80 | +0.0 % |
| 120,000 | 100,062 | 99,997 | −64 | 17.40 | 17.30 | −0.6 % |
| 200,000 | 166,660 | 166,658 | −2 | 37.47 | 37.33 | −0.4 % |
| 240,000 | 200,047 | 200,049 | +2 | 50.13 | 50.10 | −0.1 % |

| `--ctx` | A_S prefill (t/s) | L1a prefill (t/s) | Δ rate |
| ---: | ---: | ---: | ---: |
| 8,000 | 9,542 | 9,567 | +0.3 % |
| 30,000 | 8,615 | 8,532 | −1.0 % |
| 60,000 | 7,359 | 7,346 | −0.2 % |
| 120,000 | 5,751 | 5,780 | +0.5 % |
| 200,000 | 4,448 | 4,464 | +0.4 % |
| 240,000 | 3,990 | 3,993 | +0.1 % |

## One code stream reads 209.1 t/s at a 200K filler budget

`decode_ss.py --conc 1 --ctx N` puts N/1.3 filler words before the prompt, two runs per point. The probe does not record the prompt's token count, so the depth figure uses the filler budget rather than prompt tokens. Sources: `decode-{code,prose}-c1-{30,60,120,200}k.jsonl` in each directory above; the zero-filler point is the three-run c1 row from the concurrency sweep. Values below are per-stream decode tokens per second.

| `--ctx` | A_S code | L1a code | Δ code | A_S prose | L1a prose | Δ prose |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 229.5 | 241.7 | +5.3 % | 189.6 | 190.4 | +0.4 % |
| 30,000 | 226.6 | 246.5 | +8.8 % | 181.1 | 184.6 | +1.9 % |
| 60,000 | 227.9 | 235.7 | +3.4 % | 170.9 | 184.8 | +8.1 % |
| 120,000 | 211.1 | 216.2 | +2.4 % | 165.8 | 175.2 | +5.7 % |
| 200,000 | 193.3 | 209.1 | +8.2 % | 151.2 | 158.8 | +5.0 % |

On the L1a boot, code at 200K reads 13.5 % below its zero-filler rate and prose 16.6 % below. The depth runs do not use the cold-prefill protocol: their TTFT values are not cold-prefill measurements.

## Draft acceptance differs by text and depth

Acceptance is accepted draft tokens divided by drafted tokens from the server counters, reported as a fraction and taken as the median over runs. The concurrency rows use three runs and the depth rows two; the zero-filler depth row reuses c1. Sources are the same decode JSONL summaries as the throughput tables. A delta of +1.00 pp means one percentage point more acceptance.

### Acceptance against concurrency

| streams | A_S code | L1a code | Δ code | A_S prose | L1a prose | Δ prose |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6630 | 0.6790 | +1.60 pp | 0.4900 | 0.4630 | −2.70 pp |
| 2 | 0.6680 | 0.6430 | −2.50 pp | 0.4740 | 0.5030 | +2.90 pp |
| 4 | 0.6640 | 0.6560 | −0.80 pp | 0.4610 | 0.4850 | +2.40 pp |
| 6 | 0.6260 | 0.6790 | +5.30 pp | 0.4630 | 0.4600 | −0.30 pp |
| 8 | 0.6470 | 0.6520 | +0.50 pp | 0.4910 | 0.4740 | −1.70 pp |
| 12 | 0.6410 | 0.6590 | +1.80 pp | 0.4810 | 0.4810 | +0.00 pp |
| 16 | 0.6320 | 0.6340 | +0.20 pp | 0.4750 | 0.4740 | −0.10 pp |

### Acceptance at depth

| `--ctx` | A_S code | L1a code | Δ code | A_S prose | L1a prose | Δ prose |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.6630 | 0.6790 | +1.60 pp | 0.4900 | 0.4630 | −2.70 pp |
| 30,000 | 0.6885 | 0.7365 | +4.80 pp | 0.4865 | 0.4650 | −2.15 pp |
| 60,000 | 0.7340 | 0.7305 | −0.35 pp | 0.4650 | 0.5005 | +3.55 pp |
| 120,000 | 0.7310 | 0.7215 | −0.95 pp | 0.5035 | 0.5200 | +1.65 pp |
| 200,000 | 0.7340 | 0.7850 | +5.10 pp | 0.5005 | 0.5165 | +1.60 pp |
