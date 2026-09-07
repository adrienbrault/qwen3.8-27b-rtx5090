# GPU power limits (2026-09-07, results `2026-09-07-r208-power-limit/`, `-r208b-power-prefill/`, `-r208c-decode-at-300w/`)

[← all results](../RESULTS.md)

Hardware facts first: on this pair of RTX 5090s `power.min_limit` is **400 W**, so nothing below that can be set with `nvidia-smi -pl`. The two cards also ship **asymmetric defaults** — 600 W and 575 W, with max SM clocks 3135 and 3090 MHz — so a restore has to read `power.default_limit` per index rather than hardcode one number.

Measured on the live daily (MTP ns3, port 8020, unmodified) with `probes/decode_ss.py`, four arms per sweep and a repeat of the baseline as a same-config control. Draw is the busy-only mean per card (samples with `utilization.gpu` >= 50), from a continuous 250 ms `nvidia-smi` sampler joined to each row's epoch bounds.

## Decode: a 400 W cap is free

| row | default | `-pl 400` | `-lgc 0,2400` | control |
|---|---|---|---|---|
| code c1 | 216.4 tok/s | −0.7% | −12.2% | +4.0% |
| code c8 | 1,548.0 | −2.0% | −7.8% | +0.3% |
| code c16 | 2,685.9 | −2.2% | −8.5% | −1.7% |
| prose c1 @30K | 153.1 | +0.6% | −13.0% | +1.5% |

The control spans −1.7…+4.0%, so the cap's entire decode cost is inside the noise floor. The reason is that decode never reaches the cap: at default limits it draws 278/262 W per card at c1, 323/300 at c8 and 350/336 (p95 370/350) at c16 — all under 400 W.

## Prefill: the only workload above 400 W

| row | default | `-pl 400` | `-lgc 0,2100` | control |
|---|---|---|---|---|
| prefill c1 @30K | 4.31 s TTFT | +6.2% | +16.4% | +0.1% |
| prefill c1 @100K | 16.54 s | +10.0% | +21.9% | −0.1% |
| prefill c8 @30K | 20.24 s | +6.4% | +16.7% | +0.05% |

100K prefill draws 488/457 W mean and 534/512 W p95 at stock. Under the 400 W cap that falls to 378/370 W and clocks drop 2804 → 2561 MHz, for +10% TTFT. The control reproduces baseline to 0.1%, so this instrument is far tighter than the decode one and the cost is real.

## A cap and a clock lock are different levers

Clock-to-power is workload-dependent, so no single clock is "300 W" for both halves: 2100 MHz draws 318 W prefilling and ~225 W decoding; 2400 MHz draws 361 W prefilling and 250 W decoding. Taking 2100 MHz as the ~300 W setting, it costs −13.5 to −14.9% on code decode, −31.6% on prose decode at 30K, and +16 to +22% on prefill.

That is the useful result: **the clock-locked arm drew less power than the capped arm (250 vs 340 W at c8 decode) and cost roughly 4× more throughput.** A power cap is a ceiling the card only meets under sustained heavy draw, leaving short bursts free to boost; a clock lock removes the boost unconditionally, and decode is made of short bursts. To lower the power envelope, use `-pl`, not `-lgc`.

A 400 W cap on both cards is the free setting: 315 W less than the 600+575 stock sum across the pair, no measurable decode cost, ~10% slower deep prefill.

## Measurement note

`decode_ss.py` seeded prompts deterministically (`f"{c}-{i}"`), so a sweep that runs one arm per invocation sends byte-identical prompts and every arm after the first is served from the prefix cache — the first version of this sweep read 100K TTFT as 16.58 s on arm 1 and 0.93 s on arm 2, an artefact. Decode is unaffected, since steady state is sampled after prefill, but the prefill comparison was void. The probe now takes `--seed-prefix` (default empty, so older invocations are byte-identical) and each arm passes its own name.
