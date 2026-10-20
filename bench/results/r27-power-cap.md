# r27 power cap: on one boot of the served launcher, the 400 W cap reads within −3.5 to +4.7 % of stock on decode and costs 6.2 to 12.6 % of cold prefill rate (2026-10-20, results `2026-10-20-082818-r27-power-curves`)

[← all results](../RESULTS.md)

2026-10-20 08:30 to 09:03 UTC. Raw records are in [`2026-10-20-082818-r27-power-curves/`](2026-10-20-082818-r27-power-curves/) and the driver is [`scripts/r27-power-curves.sh`](../../scripts/r27-power-curves.sh).

The driver booted the served launcher [`scripts/serve-v031-nbd-daily.sh`](../../scripts/serve-v031-nbd-daily.sh) once, on the serving port. Its [boot log](2026-10-20-082818-r27-power-curves/boot-1.log) reads:

- the `pcie_ipc` all-reduce on (`pcie_ipc=1`) and batch-sharded sampling on;
- `MTP head ns3 no-block-drop` and 96 of 96 BA4 rank-layers active;
- a pool of 1,391,795 tokens and 3,883 MiB free;
- memory clock offset +4500 and core offset 0.

On that one boot it ran the [r27-ba4-curves](r27-ba4-curves.md) instruments twice. The first pass ran with both cards capped at 400 W, the serving host's limit while it serves. The second pass ran at the stock limits of 600 / 575 W, at which every other number in this repository was measured. The [audit log](2026-10-20-082818-r27-power-curves/audit.log) records the limits read back before each pass, zero engine errors, zero preemptions and zero GPU Xid events. Each pass used its own prompt seeds, so neither reused the other's prefixes.

One boot is the limit of this comparison. Both passes share one compile artifact and one boot, so boot-to-boot variation cancels. Run-to-run spread within a pass does not cancel, and the stock pass always ran second, after 15 minutes of load.

## Decode moves within the run spread

[`decode_ss.py`](../../scripts/decode_ss.py) forces 1,024 tokens per stream, sampled at temperature 0.6, three runs per shape, as in r27-ba4-curves. Values are the median of the runs in tokens per second. Aggregate is the measured sum of the stream rates. Sources: `decode-{code,prose}-{cap,stock}.jsonl`.

### Code

| streams | stock aggregate | 400 W aggregate | Δ | stock per stream | 400 W per stream | stock accept | 400 W accept |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 286.0 | 276.6 | −3.3 % | 286.0 | 276.6 | 0.716 | 0.685 |
| 2 | 471.7 | 493.9 | +4.7 % | 235.8 | 246.9 | 0.676 | 0.719 |
| 4 | 933.5 | 907.5 | −2.8 % | 233.4 | 226.9 | 0.677 | 0.638 |
| 6 | 1,323.1 | 1,378.8 | +4.2 % | 220.5 | 229.8 | 0.669 | 0.715 |
| 8 | 1,627.5 | 1,570.5 | −3.5 % | 203.4 | 196.3 | 0.647 | 0.613 |
| 12 | 2,268.3 | 2,312.5 | +1.9 % | 189.0 | 192.7 | 0.657 | 0.679 |
| 16 | 2,650.2 | 2,688.4 | +1.4 % | 165.6 | 168.0 | 0.624 | 0.638 |

### Prose

| streams | stock aggregate | 400 W aggregate | Δ | stock per stream | 400 W per stream | stock accept | 400 W accept |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 228.3 | 221.7 | −2.9 % | 228.3 | 221.7 | 0.504 | 0.482 |
| 2 | 377.6 | 379.3 | +0.5 % | 188.8 | 189.7 | 0.476 | 0.480 |
| 4 | 745.9 | 756.2 | +1.4 % | 186.5 | 189.1 | 0.470 | 0.480 |
| 6 | 1,060.8 | 1,075.5 | +1.4 % | 176.8 | 179.2 | 0.473 | 0.483 |
| 8 | 1,387.6 | 1,340.3 | −3.4 % | 173.4 | 167.5 | 0.503 | 0.474 |
| 12 | 1,822.5 | 1,810.1 | −0.7 % | 151.9 | 150.8 | 0.463 | 0.462 |
| 16 | 2,274.7 | 2,273.3 | −0.1 % | 142.2 | 142.1 | 0.488 | 0.488 |

The deltas take both signs, from −3.5 to +4.7 %, and move with draft acceptance: every cell where the capped pass reads more than 1 % slower also reads lower acceptance. The runs of the two passes overlap at the cells that read slower.

| cell | stock runs | 400 W runs |
| --- | --- | --- |
| code, 1 stream | 276.8 to 290.3 t/s | 217.4 to 302.4 t/s |
| code, 8 streams | 1,535.1 to 1,636.8 t/s | 1,523.1 to 1,648.4 t/s |

From 1 to 16 streams, no cell separates the cap from run-to-run variation on this boot. The power table below shows why: decode at no filler draws less than 400 W per card in both passes.

### One stream at depth

`decode_ss.py --conc 1 --ctx N` puts N/1.3 filler words before the prompt, two runs per point, values in per-stream tokens per second. Sources: `decode-{code,prose}-c1-{30,60,120,200}k-{cap,stock}.jsonl`.

| `--ctx` | stock code | 400 W code | Δ code | stock prose | 400 W prose | Δ prose |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30,000 | 276.9 | 269.9 | −2.5 % | 232.3 | 228.8 | −1.5 % |
| 60,000 | 259.7 | 257.6 | −0.8 % | 218.6 | 210.3 | −3.8 % |
| 120,000 | 246.9 | 238.9 | −3.2 % | 199.7 | 200.8 | +0.6 % |
| 200,000 | 211.0 | 216.8 | +2.7 % | 179.7 | 175.3 | −2.4 % |

At 60K filler and deeper, card 0 draws 412 to 460 W at stock and is held at 388 to 400 W by the cap (power table below). Six of the eight depth cells read slower under the cap, by 0.8 to 3.8 %, and two read faster, so a cost of a few percent at depth is possible. Two runs per point on one boot cannot size it.

## Cold prefill loses 6.2 to 12.6 %

[`kv_capacity_probe.py --conc 1 --tokens 1`](../../scripts/kv_capacity_probe.py) sends three fresh-seeded prompts per length, one at a time, with one output token. Latencies are recorded to 0.1 s. The prefill rate is the mean of the three `prompt_tokens / latency` rates. Sources: `prefill-{cap,stock}.jsonl`.

| `--ctx` | mean prompt tokens (stock) | stock TTFT (s) | 400 W TTFT (s) | stock prefill (t/s) | 400 W prefill (t/s) | Δ rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,000 | 6,698 | 0.70 | 0.80 | 9,569 | 8,366 | −12.6 % |
| 30,000 | 24,954 | 2.93 | 3.13 | 8,509 | 7,985 | −6.2 % |
| 60,000 | 50,029 | 6.80 | 7.30 | 7,357 | 6,845 | −7.0 % |
| 120,000 | 100,001 | 17.30 | 18.93 | 5,780 | 5,282 | −8.6 % |
| 200,000 | 166,605 | 37.20 | 41.23 | 4,479 | 4,044 | −9.7 % |
| 240,000 | 200,019 | 50.10 | 55.60 | 3,992 | 3,599 | −9.9 % |

Every capped latency is longer than every stock latency at the same length. The three prompts per length agree to the 0.1 s recording step. The 8K row is 0.7 against 0.8 s on all three prompts, one recording step, so its −12.6 % carries that resolution. From 25K to 200K prompt tokens the cap costs 6.2 to 9.9 % of prefill rate, which is 0.2 s at 25K tokens and 5.5 s at 200K. The [stock-limit prefill on 2026-09-07](r208-gpu-power-limits.md) found the same direction on an earlier configuration.

## GPU draw per pass

Each [`power-summary.json`](2026-10-20-082818-r27-power-curves/power-cap/power-summary.json) holds NVML energy per measured decode window, both cards. The [stock file](2026-10-20-082818-r27-power-curves/power-stock/power-summary.json) also lists the cap pass's windows marked `valid: false`, because that pass's samples had already been moved to its own directory.

The sampler is the serving host's NVML and RAPL logger, which wraps `decode_ss.py`'s measured windows; it is not part of this repository's `decode_ss.py`. Prefill windows were not metered. Each value is the median over the shape's runs of the sum of both cards, with card 0 / card 1 in watts in parentheses.

| shape | code, stock | code, 400 W | prose, stock | prose, 400 W |
| --- | ---: | ---: | ---: | ---: |
| 1 stream | 686 (362 / 324) | 664 (347 / 317) | 682 (358 / 323) | 678 (357 / 321) |
| 2 streams | 629 (330 / 299) | 617 (323 / 294) | 627 (331 / 296) | 626 (331 / 295) |
| 4 streams | 647 (342 / 306) | 643 (339 / 304) | 652 (344 / 307) | 652 (343 / 309) |
| 6 streams | 655 (340 / 315) | 657 (343 / 315) | 658 (344 / 315) | 660 (347 / 312) |
| 8 streams | 663 (347 / 317) | 661 (347 / 316) | 665 (350 / 314) | 665 (348 / 318) |
| 12 streams | 718 (377 / 342) | 720 (381 / 338) | 721 (378 / 345) | 720 (381 / 338) |
| 16 streams | 718 (374 / 343) | 717 (377 / 341) | 722 (379 / 342) | 718 (378 / 340) |
| 1 stream, 30K filler | 750 (395 / 355) | 722 (377 / 345) | 736 (389 / 347) | 716 (372 / 344) |
| 1 stream, 60K filler | 780 (412 / 367) | 746 (390 / 356) | 777 (415 / 363) | 744 (388 / 356) |
| 1 stream, 120K filler | 831 (441 / 390) | 769 (398 / 371) | 828 (437 / 391) | 771 (400 / 372) |
| 1 stream, 200K filler | 878 (460 / 418) | 784 (396 / 388) | 857 (443 / 414) | 775 (395 / 381) |

From 1 to 16 streams at no filler, no card reaches 400 W in either pass, and the two passes draw within 22 W of each other. At 120K and 200K filler the cap holds card 0 at 395 to 400 W against 437 to 460 W at stock.

These summaries group the one-stream windows by shape. The first three one-stream windows of each kind have no filler, and the next eight carry 30K to 200K tokens of filler. A median over all 22 one-stream windows of the stock pass reads 777 W (412 / 363). That median sits at the 60K-filler windows; it is not the no-filler draw, which is 686 W on code and 682 W on prose.

[r27-ba4-curves](r27-ba4-curves.md) (2026-10-19, results `2026-10-19-233406-r27-ba4-curves`, the BA4 launcher at stock limits) read 648 and 669 W for the no-filler one-stream code and prose windows. That is 38 and 13 W below this stock pass. Its one-stream code windows were the first measurements after its boot, while this stock pass ran right after 15 minutes of capped load. The logger records no GPU temperature, so the difference is not attributed. From 2 to 16 streams the two runs agree within 16 W.
